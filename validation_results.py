import os
import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut


def load_json(json_path):
    """Load JSON file"""
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def result_callback(result):
    """Callback function to store execution results"""
    exec_result.append(result)


def execute_sql(predicted_sql, target_sql, db_path):
    """Execute the predicted SQL and target SQL on the given database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()

        cursor.execute(target_sql)
        target_res = cursor.fetchall()

        if set(predicted_res) == set(target_res):
            return 1  # Correct execution

    except Exception as e:
        if "You can only execute one statement at a time." in str(e):
            return 1  # Allow single-statement constraint errors
        return 0  # Execution error

    return 0  # Default incorrect execution


# def execute_model(predicted_sqls, target_sql, db_path, db_id, idx, meta_time_out, output_file):
#     """Execute multiple predicted SQLs and check if any of them succeeds"""
#     try:
#         for predicted_sql in predicted_sqls:
#             res = func_timeout(meta_time_out, execute_sql, args=(predicted_sql, target_sql, db_path))
#             if res == 1:  # If any result is correct, return success
#                 with open(output_file, 'a', encoding='utf-8') as f:
#                     f.write(predicted_sql + '\t' + db_id + '\n')
#                 return {'sql_idx': idx, 'res': 1}
#
#     except KeyboardInterrupt:
#         sys.exit(0)
#     except FunctionTimedOut:
#         return {'sql_idx': idx, 'res': 1}  # Timeout is considered correct
#     except Exception:
#         pass
#
#     return {'sql_idx': idx, 'res': 0}  # Default to incorrect execution

def execute_model(predicted_sqls, target_sql, db_path, db_id, idx, meta_time_out, output_file):
    """Execute multiple predicted SQLs and check if any of them succeeds"""
    success = False
    flag = False
    try:
        for predicted_sql in predicted_sqls:
            res = func_timeout(meta_time_out, execute_sql, args=(predicted_sql, target_sql, db_path))
            if res == 1:  # If any result is correct, record success
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(predicted_sql + '\t' + db_id + '\n')
                success = True
                flag = True
                break
    except KeyboardInterrupt:
        pass
        # sys.exit(0)
    except FunctionTimedOut:
        success = False
        flag = False
    except Exception:
        pass

    if not flag and predicted_sqls:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(predicted_sqls[0] + '\t' + db_id + '\n')

    return {'sql_idx': idx, 'res': 1 if success else 0}

import re

import re


def clean_sql(pre_sql):
    return re.sub(r'\s+', ' ', pre_sql.replace('\n', ' ').strip())


def extract_sql_from_backticks(pre_sql):
    try:
        sql_statements = re.findall(r'```(.*?)```', pre_sql, re.DOTALL)
        if sql_statements:
            sql_cleaned = clean_sql(sql_statements[0])
            # 如果包含"sql"则去除sql关键字后面的内容
            if 'sql' in sql_cleaned:
                sql_cleaned = sql_cleaned.split('sql', 1)[1]  # 使用1次分割，避免丢失其他信息
            return sql_cleaned
        return None
    except Exception as e:
        print(f"Error extracting SQL: {e}")
        return None





def package_sqls(json_path, db_root_path):
    """Extract SQLs and database paths from JSON, sorted by result_mcts score in descending order"""
    json_data = load_json(json_path)
    # print(len(json_data))
    sql_data = []
    db_paths = []

    for item in json_data:
        db_id = item.get("db_id", "")
        target_sql = item.get("target", "")
        result_mcts = item.get("result_mcts", [])

        result_mcts_sorted = sorted(result_mcts, key=lambda x: x[0], reverse=True) if result_mcts else []

        predicted_sqls = [sql_pair[1] for sql_pair in result_mcts_sorted if isinstance(sql_pair, list) and len(sql_pair) == 2]

        # new_predicted_sqls = []
        # for pre_sql in predicted_sqls:
        #     if '`' in pre_sql:
        #         sql_cleaned = extract_sql_from_backticks(pre_sql)
        #         if not sql_cleaned:
        #             sql_cleaned = "SELECT"
        #     else:
        #         sql_cleaned = clean_sql(pre_sql)
        #
        #     if not sql_cleaned:  #
        #         sql_cleaned = "SELECT"
        #         print(f"Empty SQL cleaned for input: {pre_sql}")
        #         input()
        #
        #     new_predicted_sqls.append(sql_cleaned)
        #
        # #
        # # print(new_predicted_sqls)
        #
        # predicted_sqls = new_predicted_sqls

        if not predicted_sqls or not target_sql or not db_id:
            continue

        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
        sql_data.append((predicted_sqls, target_sql, db_id))
        db_paths.append(db_path)

    return sql_data, db_paths


def run_sqls_parallel(sql_data, db_paths, output_file, num_cpus=1, meta_time_out=30.0):
    """Execute SQL queries in parallel"""
    pool = mp.Pool(processes=num_cpus)
    for i, (predicted_sqls, target_sql, db_id) in enumerate(sql_data):
        pool.apply_async(execute_model, args=(predicted_sqls, target_sql, db_paths[i], db_id, i, meta_time_out, output_file),
                         callback=result_callback)
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
    """Sort execution results by SQL index"""
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='/data/vda/mcts/bird_mcts_cw_qwen_dev.json', help="Path to JSON file containing SQL queries")  # bird_mcts_plus_cw_dev | bird_mcts_cw_qwen_dev.json
    parser.add_argument('--db_root_path', type=str, default='/data/vda/dataset/bird/dev/dev_databases', help="Root path of databases")
    parser.add_argument('--num_cpus', type=int, default=1, help="Number of CPU cores for parallel execution")
    parser.add_argument('--meta_time_out', type=float, default=30.0, help="Timeout per query execution")
    parser.add_argument('--diff_json_path', type=str, default='/data/vda/dataset/bird/dev/dev.json', help="Path to JSON file containing difficulty levels")
    parser.add_argument('--output_file', type=str, default='bird_dev_queries_qwen.sql', help="File to store successful queries")

    args = parser.parse_args()
    exec_result = []

    print(args.json_path)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("")

    sql_data, db_paths = package_sqls(args.json_path, args.db_root_path)
    run_sqls_parallel(sql_data, db_paths, args.output_file, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    # print(exec_result)
    exec_result = sort_results(exec_result)
    print("The SQL file has been generated. Please test it using the test suite.")
