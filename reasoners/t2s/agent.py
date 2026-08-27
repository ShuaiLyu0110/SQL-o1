"""Text-to-SQL world model and the SQL-o1 search configuration.

This module implements the paper-specific parts of SQL-o1: semantic SQL
actions, log-likelihood self rewards, confidence based dynamic pruning, and
online execution-agreement rewards.  The generic tree traversal lives in
``reasoners.algorithm.mcts``.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from typing import Any, NamedTuple, Sequence

import requests

from reasoners import LanguageModel, SearchConfig, WorldModel


AgentAction = str

CLAUSE_KEYWORDS = [
    "select",
    "from",
    "where",
    "group by",
    "having",
    "order by",
    "limit",
    "intersect",
    "union all",
    "union",
    "except",
]
JOIN_KEYWORDS = ["right join", "inner join", "left join", "join", "on", "as"]
OTHER_KEYWORDS = ["distinct"]
BIRD_KEYWORDS = [
    "if",
    "else",
    "datediff",
    "over",
    "instr",
    "case",
    "partition by",
    "iif",
    "float",
    "real",
    "when",
    "int",
    "using",
    "timestampdiff",
    "then",
    "substr",
    "cast",
    "integer",
    "strftime",
    "end",
]
WHERE_OPS = ["not", "between", "in", "like", "is", "exists", "not null", "null"]
AGG_OPS = ["max", "min", "count", "sum", "avg"]
COND_OPS = ["and", "or"]
ORDER_OPS = ["desc", "asc"]

# Actions end at top-level semantic boundaries. Longer keywords must be tested
# first (e.g. UNION ALL before UNION and GROUP BY before GROUP).
ACTION_BOUNDARIES = sorted(
    set(CLAUSE_KEYWORDS + JOIN_KEYWORDS + COND_OPS), key=len, reverse=True
)
_BOUNDARY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(k) for k in ACTION_BOUNDARIES) + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)


class AgentState(NamedTuple):
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: AgentAction


def _join_sql(prefix: str, action: str) -> str:
    prefix = prefix.strip()
    action = action.strip()
    if not prefix:
        return action
    if not action:
        return prefix
    if action == ";":
        return prefix.rstrip() + ";"
    return f"{prefix} {action}".strip()


def _strip_model_wrapping(text: str) -> str:
    """Extract a SQL-looking response without changing SQL literals."""
    text = text.strip()
    fenced = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced[-1].strip()
    # Stop after the first statement. SQL-o1 synthesizes one read-only query.
    semicolon = text.find(";")
    if semicolon >= 0:
        text = text[: semicolon + 1]
    return re.sub(r"\s+", " ", text).strip()


def _scan_top_level_boundaries(sql: str) -> list[int]:
    """Return keyword offsets outside quotes/comments and nested expressions."""
    positions: list[int] = []
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(sql):
        char = sql[i]
        if quote:
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            i += 1
            continue
        if sql.startswith("--", i):
            newline = sql.find("\n", i + 2)
            i = len(sql) if newline < 0 else newline + 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = len(sql) if end < 0 else end + 2
            continue
        if char == "(":
            depth += 1
            i += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            match = _BOUNDARY_RE.match(sql, i)
            if match:
                positions.append(i)
                i = match.end()
                continue
        i += 1
    return positions


def segment_sql_action(completion: str) -> str:
    """Take one syntax-aligned SQL block from a model completion."""
    completion = completion.strip()
    if not completion:
        return ""
    positions = _scan_top_level_boundaries(completion)
    # The first keyword normally begins the current block; the second begins
    # the next action. If a completion starts with an expression, the first
    # keyword is the next boundary.
    start_is_boundary = bool(positions and positions[0] == 0)
    split_idx = positions[1] if start_is_boundary and len(positions) > 1 else None
    if not start_is_boundary and positions:
        split_idx = positions[0]
    action = completion if split_idx is None else completion[:split_idx]
    return action.strip()


def sql_token_similarity(left: str, right: str) -> float:
    """Jaccard similarity over SQL tokens for redundant-beam filtering."""
    token_pattern = r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<>|!=|<=|>=|[-+*/%=<>]"
    left_tokens = set(re.findall(token_pattern, left.casefold()))
    right_tokens = set(re.findall(token_pattern, right.casefold()))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _normalise_cell(value: Any) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return ("float", format(value, ".15g"))
    return (type(value).__name__, str(value))


def execute_sql_readonly(
    db_path: str,
    sql: str,
    *,
    timeout: float = 5.0,
    max_rows: int = 10_000,
) -> tuple[bool, tuple[tuple[tuple[str, str], ...], ...] | None, str | None]:
    """Execute one query safely and return a stable, hashable result.

    The SQLite connection is opened in read-only/query-only mode. Results keep
    row order (important for ORDER BY) and are capped to avoid unbounded memory
    use during search.
    """
    statement = sql.strip().rstrip(";").strip()
    if not re.match(r"^(?:select|with)\b", statement, flags=re.IGNORECASE):
        return False, None, "only SELECT/WITH queries are allowed"
    if not db_path or not os.path.isfile(db_path):
        return False, None, f"database not found: {db_path}"

    deadline = time.monotonic() + timeout
    connection: sqlite3.Connection | None = None
    try:
        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=timeout)
        connection.execute("PRAGMA query_only = ON")

        def timed_out() -> int:
            return int(time.monotonic() >= deadline)

        connection.set_progress_handler(timed_out, 10_000)
        cursor = connection.execute(statement)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            rows = rows[:max_rows]
            rows.append(("__SQL_O1_RESULT_TRUNCATED__",))
        normalised = tuple(tuple(_normalise_cell(cell) for cell in row) for row in rows)
        return True, normalised, None
    except Exception as exc:  # SQLite exposes several driver-specific subclasses.
        return False, None, str(exc)
    finally:
        if connection is not None:
            connection.close()


class AgentWorldModel(WorldModel):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        max_steps: int = 4,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> AgentState:
        return AgentState(0, "", "", "")

    def step(self, state: AgentState, action: AgentAction) -> AgentState:
        blocks_state = _join_sql(state.blocks_state, action)
        return AgentState(
            step_idx=state.step_idx + 1,
            last_blocks_state=state.blocks_state,
            blocks_state=blocks_state,
            buffered_action=action,
        )

    def is_terminal(self, state: AgentState) -> bool:
        return state.blocks_state.rstrip().endswith(";") or state.step_idx >= self.max_steps


class AgentConfig(SearchConfig):
    """Paper-specific policy/reward adapter used by generic MCTS."""

    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        batch_size: int = 1,
        reward_temperature: float = 0.6,
        base_reward: float = 100.0,
        reward_balance: float = 0.5,
        pruning_lambda: float = 0.9,
        pruning_transition: int | None = None,
        similarity_threshold: float = 0.7,
        request_timeout: float = 120.0,
        execution_timeout: float = 5.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_temperature = reward_temperature
        self.base_reward = base_reward
        self.reward_balance = reward_balance
        self.pruning_lambda = pruning_lambda
        self.pruning_transition = pruning_transition or max(1, prompt["depth_limit"] // 2)
        self.similarity_threshold = similarity_threshold
        self.request_timeout = request_timeout
        self.execution_timeout = execution_timeout
        self._observed_results: set[tuple[tuple[tuple[str, str], ...], ...]] = set()
        self._session = requests.Session()

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
        self._observed_results.clear()

    def _prompt_with_prefix(self, prefix: str) -> str:
        marker = "The incomplete SQL query:\n"
        base_prompt = self.example["input"]
        if marker in base_prompt:
            return base_prompt.replace(marker, marker + prefix, 1)
        return f"{base_prompt.rstrip()}\n{marker}{prefix}"

    def _post(self, endpoint: str, payload: dict) -> Any:
        response = self._session.post(endpoint, json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        return response.json()

    def _generate_candidates(self, model_input: str) -> list[str]:
        payload = {
            "input": model_input,
            "output": [],
            "beam_size": int(self.prompt["beam_size"]),
        }
        data = self._post(self.base_model["select"], payload)
        if isinstance(data, dict) and "candidates" in data:
            candidates = data["candidates"]
            return [item["text"] if isinstance(item, dict) else str(item) for item in candidates]
        # Compatibility with the original endpoint, which serialized a mapping
        # of generated text to its beam score.
        if isinstance(data, dict):
            return list(data.keys())
        if isinstance(data, list):
            return [item[0] if isinstance(item, (list, tuple)) else str(item) for item in data]
        raise TypeError(f"Unexpected generation response: {type(data).__name__}")

    def _score_log_likelihoods(self, model_input: str, outputs: Sequence[str]) -> list[float]:
        if not outputs:
            return []
        data = self._post(
            self.base_model.get("reward", self.base_model["select"]),
            {"input": model_input, "output": list(outputs)},
        )
        if isinstance(data, dict) and "log_likelihoods" in data:
            return [float(value) for value in data["log_likelihoods"]]
        if isinstance(data, list):
            # Old servers returned beta + log p. Undo that transformation so
            # the client remains the single owner of Eq. (4).
            return [float(value) - self.base_reward for value in data]
        raise TypeError(f"Unexpected scoring response: {type(data).__name__}")

    def _self_rewards(self, model_input: str, outputs: Sequence[str]) -> list[float]:
        log_likelihoods = self._score_log_likelihoods(model_input, outputs)
        return [self.base_reward + self.reward_temperature * value for value in log_likelihoods]

    @staticmethod
    def _continuation(full_or_suffix: str, prefix: str) -> str:
        candidate = _strip_model_wrapping(full_or_suffix)
        prefix = re.sub(r"\s+", " ", prefix).strip()
        if not prefix:
            return candidate
        if candidate.casefold().startswith(prefix.casefold()):
            return candidate[len(prefix) :].lstrip()
        # A response beginning with SELECT/WITH is a complete query. Recover a
        # continuation only if its normalized prefix actually matches.
        if re.match(r"^(?:select|with)\b", candidate, flags=re.IGNORECASE):
            prefix_tokens = prefix.split()
            candidate_tokens = candidate.split()
            if [t.casefold() for t in candidate_tokens[: len(prefix_tokens)]] == [
                t.casefold() for t in prefix_tokens
            ]:
                return " ".join(candidate_tokens[len(prefix_tokens) :])
            return ""
        return candidate

    @staticmethod
    def dynamic_threshold(scores: Sequence[float], depth: int, transition: int, scale: float) -> float:
        if not scores:
            return math.inf
        if depth <= transition:
            return scale * (sum(scores) / len(scores))
        return max(scores)

    def _prune_and_filter(self, candidates: list[tuple[str, float]], depth: int) -> list[tuple[str, float]]:
        candidates.sort(key=lambda item: item[1], reverse=True)
        threshold = self.dynamic_threshold(
            [score for _, score in candidates], depth, self.pruning_transition, self.pruning_lambda
        )
        retained = [(action, score) for action, score in candidates if score >= threshold]
        # Numeric edge cases (or negative, uncalibrated legacy scores) must not
        # erase every branch. The highest-confidence beam is always retained.
        if not retained and candidates:
            retained = [candidates[0]]

        diverse: list[tuple[str, float]] = []
        for candidate in retained:
            if all(
                sql_token_similarity(candidate[0], kept[0]) < self.similarity_threshold
                for kept in diverse
            ):
                diverse.append(candidate)
            if len(diverse) >= int(self.prompt["action_topk"]):
                break
        return diverse

    def get_actions(self, state: AgentState) -> list[tuple[AgentAction, float]]:
        model_input = self._prompt_with_prefix(state.blocks_state)
        completions = self._generate_candidates(model_input)
        actions: list[str] = []
        seen: set[str] = set()
        for completion in completions:
            suffix = self._continuation(completion, state.blocks_state)
            action = segment_sql_action(suffix)
            key = re.sub(r"\s+", " ", action).strip().casefold()
            if action and key not in seen:
                seen.add(key)
                actions.append(action)
        rewards = self._self_rewards(model_input, actions)
        return self._prune_and_filter(list(zip(actions, rewards)), state.step_idx + 1)

    def fast_reward(self, state: AgentState, action: tuple[AgentAction, float]) -> tuple[float, dict]:
        return action[1], {"intuition": action[1]}

    def reward(self, state: AgentState, action: AgentAction, intuition: float | None = None) -> tuple[float, dict]:
        # Terminal scoring belongs to rollout_reward so that Eq. (7) is applied
        # once to the complete trajectory rather than independently per edge.
        value = float(intuition if intuition is not None else 0.0)
        return value, {"self_reward": value}

    def rollout_reward(self, path, process_rewards: Sequence[float]) -> tuple[float, dict]:
        """Compute Eq. (7) and update online execution-agreement memory."""
        if not path or path[-1].state is None:
            return -math.inf, {"error": "rollout has no terminal state"}
        sql = path[-1].state.blocks_state.strip()
        process_reward = sum(process_rewards) / len(process_rewards) if process_rewards else 0.0
        terminal_reward = self._self_rewards(self._prompt_with_prefix(""), [sql])[0] if sql else 0.0

        db_path = self.example.get("db_path", "")
        succeeded, normalised_result, execution_error = execute_sql_readonly(
            db_path, sql, timeout=self.execution_timeout
        )
        if not succeeded:
            execution_reward = -1.0
        elif normalised_result and normalised_result in self._observed_results:
            execution_reward = 1.0
        else:
            execution_reward = 0.0
        if succeeded and normalised_result:
            self._observed_results.add(normalised_result)

        rollout_return = self.reward_balance * process_reward + (1.0 - self.reward_balance) * (
            terminal_reward + execution_reward
        )
        return rollout_return, {
            "process_reward": process_reward,
            "terminal_reward": terminal_reward,
            "execution_reward": execution_reward,
            "execution_error": execution_error,
            "sql": sql,
        }


from reasoners.algorithm.mcts import MCTSNode  # noqa: E402  (visualization-only imports)
from reasoners.visualization import visualize, visualize_out, visualize_save  # noqa: E402
from reasoners.visualization.tree_snapshot import EdgeData, NodeData  # noqa: E402


def blocksworld_node_data_factory(node: MCTSNode) -> NodeData:
    return NodeData(
        {
            "block state": node.state.blocks_state if node.state else "Not expanded",
            "Q": node.Q,
            "intuition": node.fast_reward_details.get("intuition", "N/A"),
            "# visited": len(node.cum_rewards),
        }
    )


def blocksworld_edge_data_factory(node: MCTSNode) -> EdgeData:
    return EdgeData({"action": node.action})


def visualize_mcts(result_rap):
    return visualize(
        result_rap,
        node_data_factory=blocksworld_node_data_factory,
        edge_data_factory=blocksworld_edge_data_factory,
    )


def visualize_mcts_save(result_rap):
    return visualize_save(
        result_rap,
        node_data_factory=blocksworld_node_data_factory,
        edge_data_factory=blocksworld_edge_data_factory,
    )


def visualize_mcts_out(data):
    return visualize_out(data)
