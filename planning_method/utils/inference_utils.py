from utils.constants import CLAUSE_KEYWORDS, SQL_KEYWORDS
from utils.normalize_sql import _is_whitespace

import sqlparse


def segment_step(sql_completion):
    parse = sqlparse.parse(sql_completion)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        (token.value.upper() if token.value.lower() in SQL_KEYWORDS else token.value)
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
