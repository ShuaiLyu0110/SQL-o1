"""Run the revised SQL-o1 MCTS inference pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from reasoners import Reasoner
from reasoners.algorithm import MCTS
from reasoners.t2s.agent import AgentConfig, AgentWorldModel


DATASET_DEFAULTS = {
    "spider": {"mcts_iters": 6, "depth_limit": 8, "exploration_weight": 0.7, "pruning_transition": 4},
    "spider_syn": {"mcts_iters": 6, "depth_limit": 8, "exploration_weight": 0.7, "pruning_transition": 4},
    "spider_real": {"mcts_iters": 6, "depth_limit": 8, "exploration_weight": 0.7, "pruning_transition": 4},
    "spider_DK": {"mcts_iters": 6, "depth_limit": 8, "exploration_weight": 0.7, "pruning_transition": 4},
    "spider_test": {"mcts_iters": 6, "depth_limit": 8, "exploration_weight": 0.7, "pruning_transition": 4},
    "bird": {"mcts_iters": 8, "depth_limit": 12, "exploration_weight": 0.8, "pruning_transition": 6},
}

DEFAULT_INPUTS = {
    "bird": "SQL-o1_bird_dev_db_id_0.json",
    "spider": "SQL-o1_spider_dev_db_id_0.json",
    "spider_syn": "SQL-o1_syn_spider_db_id_0.json",
    "spider_real": "SQL-o1_real_spider_dev_db_id_0.json",
    "spider_DK": "SQL-o1_DK_spider_dev_db_id_0.json",
    "spider_test": "SQL-o1_spider_test_db_id_0.json",
}


def dump_json(obj, filename: str) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False)


def resolve_db_path(row: dict, task_name: str, dataset_root: str) -> str:
    if row.get("db_path"):
        return os.path.abspath(row["db_path"])
    db_id = row.get("db_id")
    if not db_id:
        return ""
    root = Path(dataset_root)
    if task_name == "bird":
        path = root / "bird" / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite"
    elif task_name == "spider_test":
        path = root / "spider" / "test_database" / db_id / f"{db_id}.sqlite"
    else:
        path = root / "spider" / "database" / db_id / f"{db_id}.sqlite"
    return str(path.resolve())


def _rollout_log(result) -> list[dict]:
    logs = []
    seen_sql = set()
    for rollout_return, path in result.trace_in_each_iter or []:
        terminal = path[-1]
        sql = terminal.state.blocks_state if terminal.state is not None else ""
        if sql in seen_sql:
            continue
        seen_sql.add(sql)
        logs.append(
            {
                "return": float(rollout_return),
                "sql": sql,
                "details": terminal.rollout_details,
            }
        )
    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQL-o1 self-reward heuristic search")
    parser.add_argument("--task_name", choices=DATASET_DEFAULTS, default="spider")
    parser.add_argument("--input_file", default="", help="Preprocessed JSON; uses the task default when empty")
    parser.add_argument("--output_file", default="", help="MCTS result JSON")
    parser.add_argument("--dataset_root", default="./dataset", help="Root containing spider/ and bird/")
    parser.add_argument("--api_url", default="http://localhost:8000/llm")
    parser.add_argument("--mcts_iters", type=int, default=None)
    parser.add_argument("--depth_limit", type=int, default=None)
    parser.add_argument("--exploration_weight", type=float, default=None)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--action_topk", type=int, default=3)
    parser.add_argument("--reward_temperature", type=float, default=0.6)
    parser.add_argument("--base_reward", type=float, default=100.0)
    parser.add_argument("--reward_balance", type=float, default=0.5)
    parser.add_argument("--pruning_lambda", type=float, default=0.9)
    parser.add_argument("--pruning_transition", type=int, default=None)
    parser.add_argument("--similarity_threshold", type=float, default=0.7)
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--execution_timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test example limit")
    parser.add_argument("--quiet_mcts", action="store_true")
    return parser.parse_args()


def run_text2sql(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS[args.task_name]
    mcts_iters = args.mcts_iters if args.mcts_iters is not None else defaults["mcts_iters"]
    depth_limit = args.depth_limit if args.depth_limit is not None else defaults["depth_limit"]
    exploration_weight = (
        args.exploration_weight
        if args.exploration_weight is not None
        else defaults["exploration_weight"]
    )
    pruning_transition = (
        args.pruning_transition
        if args.pruning_transition is not None
        else defaults["pruning_transition"]
    )
    if not 0.0 < args.reward_balance < 1.0:
        raise ValueError("--reward_balance must be in (0, 1)")
    if not 0.0 < args.pruning_lambda <= 1.0:
        raise ValueError("--pruning_lambda must be in (0, 1]")
    if not 1 <= pruning_transition <= depth_limit:
        raise ValueError("--pruning_transition must be between 1 and --depth_limit")

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_file = args.input_file or str(Path("dataset") / DEFAULT_INPUTS[args.task_name])
    output_file = args.output_file or str(Path("mcts_results") / f"{args.task_name}_mcts_dev.json")
    with open(input_file, encoding="utf-8") as file:
        sql_data = json.load(file)
    if args.limit is not None:
        sql_data = sql_data[: args.limit]

    base_model = {"select": args.api_url, "simulate": args.api_url, "reward": args.api_url}
    search_parameters = {
        "depth_limit": depth_limit,
        "beam_size": args.beam_size,
        "action_topk": args.action_topk,
    }
    results = []
    for source_row in tqdm(sql_data, desc=f"SQL-o1/{args.task_name}"):
        row = copy.deepcopy(source_row)
        row["db_path"] = resolve_db_path(row, args.task_name, args.dataset_root)
        world_model = AgentWorldModel(
            base_model=base_model, prompt=search_parameters, max_steps=depth_limit
        )
        config = AgentConfig(
            base_model=base_model,
            prompt=search_parameters,
            reward_temperature=args.reward_temperature,
            base_reward=args.base_reward,
            reward_balance=args.reward_balance,
            pruning_lambda=args.pruning_lambda,
            pruning_transition=pruning_transition,
            similarity_threshold=args.similarity_threshold,
            request_timeout=args.request_timeout,
            execution_timeout=args.execution_timeout,
        )
        algorithm = MCTS(
            depth_limit=depth_limit,
            disable_tqdm=args.quiet_mcts,
            output_trace_in_each_iter=True,
            n_iters=mcts_iters,
            w_exp=exploration_weight,
            calc_q=np.mean,
            output_strategy="max_value",
        )
        result = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)(row)

        final_sql = result.terminal_state.blocks_state if result.terminal_state is not None else ""
        row["result_mcts"] = _rollout_log(result)
        row["result_mcts_best"] = {
            "value": None if result.cum_reward == -np.inf else float(result.cum_reward),
            "sql": final_sql,
        }
        row["search_config"] = {
            "rollout_count": mcts_iters,
            "beam_size": args.beam_size,
            "action_space_size": args.action_topk,
            "maximum_depth": depth_limit,
            "exploration_weight": exploration_weight,
            "reward_temperature": args.reward_temperature,
            "base_reward": args.base_reward,
            "reward_balance": args.reward_balance,
            "pruning_lambda": args.pruning_lambda,
            "pruning_transition": config.pruning_transition,
            "similarity_threshold": args.similarity_threshold,
        }
        results.append(row)
        dump_json(results, output_file)


if __name__ == "__main__":
    run_text2sql(parse_args())
