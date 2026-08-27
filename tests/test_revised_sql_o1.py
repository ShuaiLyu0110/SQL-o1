import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from build_cold_start_data import build_cold_start_data
from reasoners import SearchConfig, WorldModel
from reasoners.algorithm.mcts import MCTS
from reasoners.t2s.agent import (
    AgentConfig,
    AgentState,
    execute_sql_readonly,
    segment_sql_action,
    sql_token_similarity,
)
from validation_results import package_sqls


class SQLActionTests(unittest.TestCase):
    def test_segments_at_top_level_clause(self):
        self.assertEqual(
            segment_sql_action("SELECT name FROM users WHERE age > 18;"),
            "SELECT name",
        )

    def test_does_not_split_nested_subquery(self):
        self.assertEqual(
            segment_sql_action(
                "WHERE id IN (SELECT id FROM staff WHERE active = 1) AND score > 3;"
            ),
            "WHERE id IN (SELECT id FROM staff WHERE active = 1)",
        )

    def test_dynamic_threshold_matches_manuscript(self):
        scores = [90.0, 80.0, 70.0]
        self.assertEqual(AgentConfig.dynamic_threshold(scores, 2, 4, 0.9), 72.0)
        self.assertEqual(AgentConfig.dynamic_threshold(scores, 5, 4, 0.9), 90.0)

    def test_similarity_keeps_different_columns(self):
        self.assertLess(sql_token_similarity("SELECT user.name", "SELECT user.age"), 0.7)
        self.assertEqual(sql_token_similarity("SELECT user.name", "select  user . name"), 1.0)


class ExecutionRewardTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "example.sqlite")
        connection = sqlite3.connect(self.db_path)
        connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany("INSERT INTO users VALUES (?, ?)", [(1, "Ada"), (2, "Lin")])
        connection.commit()
        connection.close()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_readonly_execution_and_mutation_rejection(self):
        ok, result, error = execute_sql_readonly(
            self.db_path, "SELECT name FROM users ORDER BY id;"
        )
        self.assertTrue(ok, error)
        self.assertEqual(len(result), 2)
        ok, _, _ = execute_sql_readonly(self.db_path, "DELETE FROM users")
        self.assertFalse(ok)

    def test_online_agreement_is_zero_then_positive(self):
        config = AgentConfig(
            base_model={"select": "unused", "reward": "unused"},
            prompt={"depth_limit": 2, "beam_size": 5, "action_topk": 3},
        )
        config.update_example({"input": "prompt", "db_path": self.db_path})
        config._self_rewards = lambda model_input, outputs: [80.0 for _ in outputs]
        root = SimpleNamespace(state=AgentState(0, "", "", ""))
        terminal = SimpleNamespace(
            state=AgentState(1, "", "SELECT name FROM users ORDER BY id;", ";")
        )
        first, first_details = config.rollout_reward([root, terminal], [90.0])
        second, second_details = config.rollout_reward([root, terminal], [90.0])
        self.assertEqual(first_details["execution_reward"], 0.0)
        self.assertEqual(second_details["execution_reward"], 1.0)
        self.assertEqual(second - first, 0.5)


class TinyWorld(WorldModel):
    def init_state(self):
        return 0

    def step(self, state, action):
        return state + action

    def is_terminal(self, state):
        return state >= 2


class TinyConfig(SearchConfig):
    def get_actions(self, state):
        return [(1, 5.0), (2, 4.0)] if state == 0 else [(1, 3.0)]

    def reward(self, state, action, intuition=None, **kwargs):
        return float(intuition), {}

    def rollout_reward(self, path, process_rewards):
        value = float(np.mean(process_rewards))
        return value, {"process_reward": value}


class MCTSTests(unittest.TestCase):
    def test_one_rollout_return_is_propagated_to_every_node(self):
        algorithm = MCTS(
            n_iters=3,
            depth_limit=2,
            w_exp=0.7,
            output_strategy="max_value",
            disable_tqdm=True,
        )
        result = algorithm(TinyWorld(), TinyConfig())
        self.assertIsNotNone(result.terminal_state)
        for child in result.tree_state.children:
            for value in child.cum_rewards:
                self.assertIsInstance(value, float)
        self.assertEqual(result.trace[1][1][0], 1)


class ColdStartDataTests(unittest.TestCase):
    def test_multimodel_failures_are_prioritized_and_ratio_is_exact(self):
        schema = [{"input": f"base-{i}", "target": f"sql-{i}"} for i in range(10)]
        failures = [
            {"input": "p-a", "target": "A", "db_id": "db", "failure_model": "m1"},
            {"input": "p-a2", "target": "A", "db_id": "db", "failure_model": "m2"},
            {"input": "p-b", "target": "B", "db_id": "db", "failure_model": "m1"},
        ]
        combined, metadata = build_cold_start_data(
            schema, failures, psg_ratio=1 / 11, max_psg_examples=None, seed=1
        )
        self.assertEqual(len(combined), 11)
        self.assertEqual(metadata["selected_psg_examples"], 1)
        self.assertIn("A", {item["target"] for item in combined})


class ResultExportTests(unittest.TestCase):
    def test_export_uses_search_choice_not_gold_or_an_oracle_rollout(self):
        with tempfile.TemporaryDirectory() as tempdir:
            result_file = Path(tempdir) / "results.json"
            result_file.write_text(
                json.dumps(
                    [
                        {
                            "db_id": "db",
                            "target": "SELECT gold;",
                            "result_mcts": [
                                {"return": 99.0, "sql": "SELECT oracle_candidate;"}
                            ],
                            "result_mcts_best": {
                                "value": 80.0,
                                "sql": "SELECT search_selected;",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            sql_data, _ = package_sqls(str(result_file), tempdir)
            self.assertEqual(sql_data[0][0], ["SELECT search_selected;"])


if __name__ == "__main__":
    unittest.main()
