"""
Code is adapted from https://github.com/shuaichenchang/prompt-text-to-sql/blob/6cbcb2f8dd82f982e0f4964098dc56fe4b7fd57c/utils.py
"""
from utils.constants import SQL_KEYWORDS, AGG_OPS

import sqlparse


def lexical(query, values):
    if isinstance(query, str):
        for placeholder, value in values.items():
            query = query.replace(placeholder, value)
    elif isinstance(query, list):
        for i in range(len(query)):
            if query[i] in values:
                query[i] = values[query[i]]
    return query


def delexical(query):
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


def format_query(q, format_type):
    if format_type == 'unnormalized':
        return q["query"]
    elif format_type == 'normalized':
        return q["gold"]["query_normalized"]
    else:
        raise ValueError(f"format_type {format_type} not supported")


def _is_whitespace(sqlparse_token):
    return sqlparse_token.ttype == sqlparse.tokens.Whitespace


def normalize_sql(sql_exp):
    sql_exp = sql_exp.replace('"', "'")
    if sql_exp.count("'") % 2 != 0:  # odd number of single quotes, meaning the value is incomplete or value contains a single quote
        odd_quotes = True
    else:
        odd_quotes = False
    
    if not odd_quotes:
        sql_exp, values = delexical(sql_exp)
        sql_exp = sql_exp.lower()
    
    sql_exp = sql_exp.rstrip(";")
    parse = sqlparse.parse(sql_exp)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
        for token in flat_tokens if not _is_whitespace(token)
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
        sql_tokens = lexical(sql_tokens, values)
        sql_lower = lexical(sql_lower, values)
    # else:
    #     print("Cannot process the following SQL")
    #     print(sql_exp, sql_tokens)

    return sql_lower

if __name__ == "__main__":
    print(
        normalize_sql(
            "select fname, lname from student where age < (select avg(age) from student where age > 0);"
        )
    )
