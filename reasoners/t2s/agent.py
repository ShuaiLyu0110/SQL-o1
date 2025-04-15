from reasoners import WorldModel, LanguageModel, SearchConfig
from typing import NamedTuple
import sqlparse
import requests
import re
AgentAction = str

CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'having', 'order by', 'limit', 'intersect', 'union', 'except', 'union all']
JOIN_KEYWORDS = ['join', 'on', 'as', 'right join', 'inner join', 'left join']
OTHER_KEYWORDS = ['distinct']
BIRD_KEYWORDS = ['if', 'else', 'datediff', 'over', 'instr', 'case', 'partition by', 'iif', 'float', 'real', 'when', 'int', 'using', 'timestampdiff', 'then', 'substr', 'cast', 'integer', 'strftime', 'end']
WHERE_OPS = ['not', 'between', 'in', 'like', 'is', 'exists', 'not null', 'null']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(OTHER_KEYWORDS)
SQL_KEYWORDS.extend(BIRD_KEYWORDS)
SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)
SQL_KEYWORDS = [i.upper() for i in SQL_KEYWORDS]

class AgentState(NamedTuple):
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: AgentAction

class AgentWorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 4,
                 batch_size: int = 1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> AgentState:
        return AgentState(step_idx=0, 
                          last_blocks_state="", 
                          blocks_state="", 
                          buffered_action="")

    def step(self, state: AgentState, action: AgentAction) -> tuple[AgentState, dict]:
        step_idx = state.step_idx
        # blocks_state = state.blocks_state + action + ("; " if action != "done" and action != "none" else "")

        if action == ";" or action == " ;" or action.endswith(";"):
            # blocks_state = state.blocks_state + ("" if state.blocks_state.endswith(";") or state.blocks_state.endswith("; ") else "; ") + action
            # blocks_state = state.blocks_state + " " + action
            blocks_state = state.blocks_state + action if not state.blocks_state else state.blocks_state + " " + action
        else:
            blocks_state = state.blocks_state + action if not state.blocks_state else state.blocks_state + " " + action

        new_buffered_action = action

        state = AgentState(step_idx=step_idx + 1,
                        last_blocks_state=state.blocks_state,
                        blocks_state=blocks_state,
                        buffered_action=new_buffered_action)
        return state

    def is_terminal(self, state: AgentState) -> bool:
        if state.buffered_action in [';', ' ;'] or state.buffered_action.endswith(";"):
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False
    
    
    
class AgentConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size: int = 1,
                 reward_alpha: float = 0.5,
                 goal_reward_default: float = 0.,
                 goal_reached_reward: float = 100.) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward

    def lexical(self, query, values):
        if isinstance(query, str):
            for placeholder, value in values.items():
                query = query.replace(placeholder, value)
        elif isinstance(query, list):
            for i in range(len(query)):
                if query[i] in values:
                    query[i] = values[query[i]]
        return query

    def delexical(self, query):
        values = {}
        new_query = ""
        in_value = False
        in_col = False
        value = ""
        placeholder_id = 0
        new_query = ""
        for char in query:
            if char == "'":
                in_value = not in_value
                value += char
                if not in_value:
                    values[f"value_{placeholder_id}"] = value
                    new_query += f"value_{placeholder_id}"
                    placeholder_id += 1
                    value = ""
            else:
                if not in_value:
                    new_query += char
                else:
                    value += char
        return new_query, values

    def format_query(self, q, format_type):
        if format_type == 'unnormalized':
            return q["query"]
        elif format_type == 'normalized':
            return q["gold"]["query_normalized"]
        else:
            raise ValueError(f"format_type {format_type} not supported")

    def _is_whitespace(self, sqlparse_token):
        return sqlparse_token.ttype == sqlparse.tokens.Whitespace



    def normalize_sql(self, sql_exp):
        sql_exp = sql_exp.replace('"', "'")
        if sql_exp.count(
                "'") % 2 != 0:  # odd number of single quotes, meaning the value is incomplete or value contains a single quote
            odd_quotes = True
        else:
            odd_quotes = False

        if not odd_quotes:
            sql_exp, values = self.delexical(sql_exp)
            sql_exp = sql_exp.lower()

        sql_exp = sql_exp.rstrip(";")
        parse = sqlparse.parse(sql_exp)
        sql = parse[0]
        flat_tokens = sql.flatten()
        sql_tokens = [
            (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
            for token in flat_tokens if not self._is_whitespace(token)
        ]

        sql_lower = ' '.join(sql_tokens)
        sql_lower = sql_lower.replace(' . ', '.')
        for op in AGG_OPS:
            sql_lower = sql_lower.replace(f" {op.upper()} (", f" {op.upper()}(")
        sql_lower = sql_lower.replace('( ', '(')
        sql_lower = sql_lower.replace(' )', ')')
        sql_lower = sql_lower.replace(' ,', ',')

        ### BIRD-SQL special cases ###
        sql_lower = sql_lower.replace(' AS text', ' AS TEXT')
        sql_lower = sql_lower.replace(' length(', ' LENGTH(')
        sql_lower = sql_lower.replace(' total(', ' TOTAL(')
        sql_lower = sql_lower.replace(' round(', ' ROUND(')
        ### END ###

        sql_lower = sql_lower.rstrip(";")
        sql_lower += ';'

        if not odd_quotes:
            # sql_tokens = self.lexical(sql_tokens, values)
            sql_lower = self.lexical(sql_lower, values)
        # else:
        #     print("Cannot process the following SQL")
        #     print(sql_exp, sql_tokens)
        return sql_lower

    def segment_step(self, sql_completion):
        try:
            parse = sqlparse.parse(sql_completion)
            sql = parse[0]
        except Exception as e:
            return ""
        flat_tokens = sql.flatten()
        sql_tokens = [
            (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
            for token in flat_tokens
        ]

        step_length = 0
        for i, token in enumerate(sql_tokens[1:]):
            if token.lower() in CLAUSE_KEYWORDS:
                step_length = i + 1
                break

        if step_length == 0:
            # No more clauses, the entire completion is a step
            return sql_completion
        else:
            return "".join(sql_tokens[:step_length])

    def get_actions(self, state: AgentState) -> list[AgentAction]:
        if state.step_idx == self.prompt['deapth_limit']-1:
            if self.example['target'].startswith(state.blocks_state):
                return [('done',100.0)]
            else:
                return [('done',99.99)]

            # if self.example['output'].startswith(state.blocks_state):
            #     return [('done',100.0)]
            # else:
            #     return [('done',99.99)]
        else:
            # output = requests.post(self.base_model['select'], json={"instruction": self.example['instruction'], "input": self.example['instruction'] + "\n" +self.example['input']+state.blocks_state, "output": [] }).json()
            # print(self.example['input'])
            print(state.blocks_state)
            print(self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state))
            # input()
            output = requests.post(self.base_model['select'], json={ "input": self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state), "output": [] }).json()

            # def is_valid_string(s):
            #     pattern = r'^(\[[^\]]+\]: <[^>]+>)'
            #     if "; " not in s:
            #         return bool(s in ['none','done'])
            #     else:
            #         if not s.endswith("; done"):
            #             return False
            #         else:
            #             #  and x.split('<')[-1].split('>')[0] in self.example['input']
            #             return all([bool(re.match(pattern, x)) for x in s.split("; ")[:-1]])

            # def is_valid_string(s):
            #     if ";" not in s:
            #         if s == "done" or s == " done":
            #             return s
            #         return ""
            #     else:
            #         if s == "; done" or s == ";done":
            #             return "; done"
            #         elif s.endswith("done"):
            #             return s.split("done")[0]
            #         else:
            #             return s

            # sql_completions = []
            # for key in output.keys():
            #     key = is_valid_string(key)
            #     if key:
            #         if key not in ["done", " done", "; done", ";done"]:
            #             sql_completions.append(self.normalize_sql(key))
            #         else:
            #             sql_completions.append(key)
            #     else:
            #         continue

            def is_valid_string(s):
                if ";" not in s:
                    return False
                else:
                    return True

            sql_completions = [key for key in output.keys() if is_valid_string(key)]
            # sql_completions = [self.normalize_sql(key) for key in output.keys() if is_valid_string(key)]

            actions = set([
                (
                    self.segment_step(sql[len(state.blocks_state):].lstrip()).rstrip()
                    if len(sql) > len(state.blocks_state)
                    else sql
                )
                for sql in sql_completions
            ])

            actions = list(actions)

            # p_reward = requests.post(self.base_model['select'], json={"input": self.example['instruction'] + "\n" + self.example['input']+state.blocks_state, "output": actions}).json()

            p_reward = requests.post(self.base_model['select'], json={"input": self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state), "output": actions}).json()
            actions_scores_list = [(a,min(r,99.99)) for a,r in zip(actions, p_reward)]
            actions_scores_list = sorted(actions_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['step_topk']]
            
            # if self.example['output'].startswith(state.blocks_state):
            #     gt_action = self.example['output'][len(state.blocks_state):]
            #     actions_scores_list = [(gt_action, 100.0)]+[(a,r) for a,r in actions_scores_list if a!=gt_action]
                # actions_scores_list = [(gt_action, requests.post(self.base_model['select'], json={ "input": self.example['input']+state.blocks_state, "output": [gt_action] }).json()[0])]+[(a,r) for a,r in actions_scores_list if a!=gt_action]        
            return actions_scores_list

    def fast_reward(self, state: AgentState, action: AgentAction) -> tuple[float, dict]:     
        intuition = action[1]
        self_eval = intuition

        return (self.calculate_reward(intuition, self_eval),
                {'intuition': intuition, "self_eval": self_eval})

    def calculate_reward(self, intuition, goal_reached=None) -> float:
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = goal_reached[1]
        else:
            goal_reward = goal_reached[1]
        return intuition * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: AgentState, action: AgentAction,
               intuition: float = None) -> tuple[float, dict]:
        # if action == "done" or action == "none" or action == " done":
        if action.endswith(";"):
            goal_reached_if = True
            # goal_reached_score = requests.post(self.base_model['reward'], json={ "input": self.example['instruction'] + "\n" + self.example['input'], "output": [state.blocks_state+action]}).json()[0]
            goal_reached_score = requests.post(self.base_model['reward'], json={ "input":self.example['input'], "output": [state.blocks_state+action]}).json()[0]

            goal_reached = (goal_reached_if, goal_reached_score)
        else:
            goal_reached = (False, 0.0)
        return (self.calculate_reward(intuition, goal_reached),
                {'intuition': intuition, 'goal_reached': goal_reached})

from reasoners.visualization import visualize,visualize_save,visualize_out
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode

# (Optional) You can write node_data_factory and edge_data_factory to show customized information.
def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData({"block state": n.state.blocks_state if n.state else "Not expanded",
                    #  "function state": '\n'.join(n.state.functions_state) if n.state else "Not expanded",
                    #  "# goals satisfied": n.reward_details["goal_reached"][1] if hasattr(n, "reward_details") else "N/A",
                     "Q": n.Q,
                     "intuition": n.fast_reward_details["intuition"] if n.id!=0 else "N/A",
                     "# visited": len(n.cum_rewards)})

def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
    return EdgeData({# "Q": n.Q,
                    #  "intuition": n.fast_reward_details["intuition"],
                    #  "self_eval": n.fast_reward_details["intuition"],
                     "action": n.action})
    
def visualize_mcts(result_rap):
    visualize(result_rap,
            node_data_factory=blocksworld_node_data_factory,
            edge_data_factory=blocksworld_edge_data_factory)   
    
def visualize_mcts_save(result_rap):
    return visualize_save(result_rap,
            node_data_factory=blocksworld_node_data_factory,
            edge_data_factory=blocksworld_edge_data_factory)  
    
def visualize_mcts_out(data):
    visualize_out(data) 

