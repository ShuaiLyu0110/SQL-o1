import os
import io
import json
import random
import numpy as np
import torch
import argparse
import logging

IGNORE_INDEX = -100
CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'having', 'order by', 'limit', 'intersect', 'union', 'except',
                   'union all']

JOIN_KEYWORDS = ['join', 'on', 'as', 'right join', 'inner join', 'left join']
OTHER_KEYWORDS = ['distinct']
BIRD_KEYWORDS = ['if', 'else', 'datediff', 'over', 'instr', 'case', 'partition by', 'iif', 'float', 'real', 'when',
                 'int', 'using', 'timestampdiff', 'then', 'substr', 'cast', 'integer', 'strftime', 'end']
WHERE_OPS = ['not', 'between', 'in', 'like', 'exists', 'not null', 'null']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(OTHER_KEYWORDS)
SQL_KEYWORDS.extend(BIRD_KEYWORDS)
# SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)


def set_seed(seed: int):
    """
    Set random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)  # Python 内置的随机数生成器
    np.random.seed(seed)  # NumPy 随机数生成器
    torch.manual_seed(seed)  # PyTorch 随机数生成器（CPU）
    torch.cuda.manual_seed(seed)  # PyTorch 随机数生成器（GPU）
    torch.cuda.manual_seed_all(seed)  # 多个 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 保证卷积操作的确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动优化，确保结果一致


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def truncate_sql_before_keywords(sql: str, keywords: list) -> str:
    """
    Truncate the SQL query randomly before one of the clause keywords (case-sensitive, for uppercase keywords).

    Args:
        sql (str): The SQL query to truncate.
        keywords (list): List of clause keywords in uppercase.

    Returns:
        str: Truncated SQL query.
    """
    # Find positions of all keywords in the SQL query
    positions = []
    for keyword in keywords:
        start_idx = 0
        while (start_idx := sql.find(keyword.upper(), start_idx)) != -1:
            positions.append(start_idx)  # Add the position before the keyword
            start_idx += len(keyword)

    # If no keywords are found, return the original SQL
    if not positions:
        return sql

    # Randomly select a truncation point
    truncation_point = random.choice([pos for pos in positions if pos != 0])

    # Truncate the SQL at the selected point
    return sql[:truncation_point].strip()


def truncate_sql_before_keywords_v2(sql: str, keywords: list) -> list:
    """
    Return all possible truncated SQL queries, truncating before one of the clause keywords.
    Ensure that the truncation respects the order from short to long keywords.

    Args:
        sql (str): The SQL query to truncate.
        keywords (list): List of clause keywords in uppercase.

    Returns:
        list: A list of all possible truncated SQL queries.
    """
    # 对关键字按长度从短到长排序，确保长关键字不在短关键字之前截断
    keywords = sorted(keywords, key=lambda k: len(k))

    # 查找所有关键字的位置
    positions = []
    for keyword in keywords:
        start_idx = 0
        while (start_idx := sql.find(keyword, start_idx)) != -1:
            positions.append((start_idx, keyword))  # 保存位置和关键字
            start_idx += len(keyword)

    # 如果没有找到任何关键字，返回原始 SQL 查询
    if not positions:
        return [sql]

    # 按照关键字位置排序，确保先截断离起始位置最近的关键字
    positions.sort()

    # 生成所有可能的截断后的 SQL 查询
    truncated_sqls = []
    for i, (pos, keyword) in enumerate(positions):
        if pos != 0:  # 如果位置不在开头，才进行截断
            truncated_sqls.append(sql[:pos].strip())

    # 添加原始 SQL 查询到列表
    truncated_sqls.append(sql)

    return truncated_sqls


def normal_process(list_data_dict: list[dict, ...], args):
    datasets_li = []
    for idx, example in enumerate(list_data_dict):
        dic_ = {}

        if args.step_by_step:
            dic_["input"] = f"{example['input']}"
            dic_["target"] = f"{example['target']}"
        else:
            dic_["instruction"] = example['question']
            dic_["input"] = example['schema'] + "\nAnswer: "
            dic_['output'] = example['sql']
            dic_['db_id'] = example['db_id']
            dic_['history'] = []



        # dic_["input"] = example['input']
        # dic_["input"] = example['schema'] + "\nAnswer: "
        # dic_['output'] = example['sql']
        # dic_['db_id'] = example['db_id']
        # dic_['history'] = []

        datasets_li.append(dic_)

    logging.warning("Storing data... This may take some time...")
    jdump(datasets_li, os.path.join(os.path.dirname(args.file),
                                    f"normal_process_{os.path.splitext(os.path.basename(args.file))[0]}.json"))


def random_truncation_process(list_data_dict: list[dict, ...], args):
    ratio = args.ratio
    total_examples = len(list_data_dict)
    num_incomplete = int(total_examples * ratio)

    # 随机选择一些索引来生成未完全的 SQL
    incomplete_indices = set(random.sample(range(total_examples), num_incomplete))
    datasets_li = []
    for idx, example in enumerate(list_data_dict):

        dic_ = {}

        # 生成 SQL（未完全或完整）
        if idx in incomplete_indices:
            # 截断 SQL 生成未完全版本


            if args.step_by_step:
                incomplete_sql = truncate_sql_before_keywords(example['target'], CLAUSE_KEYWORDS)
                dic_["input"] = f"{example['input']}" + f"{incomplete_sql}"
                dic_["target"] = f"{example['target']}"

            else:
                incomplete_sql = truncate_sql_before_keywords(example['sql'], CLAUSE_KEYWORDS)
                dic_["instruction"] = f"{example['question']}"
                source = f"{example['schema']}" + "\nAnswer: "
                source += incomplete_sql
                dic_["input"] = source
                dic_["output"] = f"{example['sql']}"
                dic_['history'] = []
            datasets_li.append(dic_)

        else:

            if args.step_by_step:
                dic_["input"] = f"{example['input']}"
                dic_["target"] = f"{example['target']}"

            else:
                dic_["instruction"] = f"{example['question']}"
                source = f"{example['schema']}" + "\nAnswer: "
                dic_["input"] = source
                dic_["output"] = f"{example['sql']}"
                dic_["history"] = []
            datasets_li.append(dic_)

    logging.warning("Storing data... This may take some time...")
    jdump(datasets_li, os.path.join(os.path.dirname(args.file),
                                    f"random_truncation_{os.path.splitext(os.path.basename(args.file))[0]}.json"))


def Progressive_Truncation(list_data_dict: list[dict, ...], args):
    ratio = args.ratio
    total_examples = len(list_data_dict)
    num_incomplete = int(total_examples * ratio)

    # 随机选择一些索引来生成未完全的 SQL
    incomplete_indices = set(random.sample(range(total_examples), num_incomplete))
    datasets_li = []
    for idx, example in enumerate(list_data_dict):
        dic_ = {}

        # 生成 SQL（未完全或完整）
        if idx in incomplete_indices:
            # 截断 SQL 生成未完全版本

            if args.step_by_step:
                incomplete_sqls = truncate_sql_before_keywords_v2(example['target'], SQL_KEYWORDS)
            else:
                incomplete_sqls = truncate_sql_before_keywords_v2(example['sql'], SQL_KEYWORDS)
            for incomplete_sql in incomplete_sqls:
                temp_ = {}

                if args.step_by_step:
                    temp_["input"] = f"{example['input']}" + f"{incomplete_sql}"
                    temp_["target"] = f"{example['target']}"

                else:
                    temp_["instruction"] = f"{example['question']}"
                    source = f"{example['schema']}" + "\nAnswer: "
                    source += incomplete_sql
                    temp_["input"] = source
                    temp_["output"] = f"{example['sql']}"
                    temp_['history'] = []
                datasets_li.append(temp_)

        else:
            if args.step_by_step:
                dic_["input"] = f"{example['input']}"
                dic_["target"] = f"{example['target']}"
            else:
                dic_["instruction"] = f"{example['question']}"
                source = f"{example['schema']}" + "\nAnswer: "
                dic_["input"] = source
                dic_["output"] = f"{example['sql']}"
                dic_["history"] = []
            datasets_li.append(dic_)

    logging.warning("Storing data... This may take some time...")
    jdump(datasets_li, os.path.join(os.path.dirname(args.file),
                                    f"progressive_truncation_{os.path.splitext(os.path.basename(args.file))[0]}.json"))


def preprocess(args):
    data_path = args.file
    process_type = args.type

    logging.warning("Loading data...")
    list_data_dict = jload(data_path)

    logging.warning("Formatting inputs...")

    if process_type == "Normal":
        normal_process(list_data_dict, args)

    elif process_type == "Random":
        random_truncation_process(list_data_dict, args)

    elif process_type == "Progressive":
        Progressive_Truncation(list_data_dict, args)


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--file", type=str, help="The name of the file to process", default="data/bird_train_plus.json")
    parser.add_argument("--type", type=str,
                        help="Processing Type (Normal | Random Truncation | Progressive Truncation)", default="Progressive")
    parser.add_argument("--step_by_step",action="store_true", help="Disable step-by-step processing. Default is True.",
                        default=True)
    parser.add_argument("--ratio", type=float, help="Truncated Sample Ratio", default=0.4)
    args = parser.parse_args()

    print(f"处理的文件：{args.file}, 类型为：{args.type}， 比例为：{args.ratio}\n")
    preprocess(args)
    logging.warning("Processing complete.")
