# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import json
import re
import sqlite3
import traceback
import os
from vllm import LLM, SamplingParams
from func_timeout import func_set_timeout
import func_timeout
import tqdm

prompt_cw_temp_sft = """Given the following database schema and question, your task is to write a valid SQL query whose execution will accurately answer the question. If the value below the incomplete SQL query is not empty, your task is to complete it into a full SQL query. Remember to end the query with a semicolom ```;```.

Database schema:
{ds}

Sample rows of each table:
{sr}

Question:
{qs}{hint}

Question hint:
{sql}

The incomplete SQL query:
{sql}

Answer the question by a SQL query only with no explanation:
"""


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print("=" * 10, e)
        return None


class LLM_Model(object):
    def __init__(self, model=''):

        self.model = model
        model = model.lower().replace('_', '').replace('-', '')
        if 'qwen2' in model:
            self.tag = 'qwen2'
        elif 'llama3' in model:
            self.tag = 'llama3'
        elif 'llama2' in model:
            self.tag = 'llam2'
        elif 'deepseek' in model:
            self.tag = 'deepseek'
        elif 'mistral' in model:
            self.tag = 'mistral'
        elif 'codellama' in model:
            self.tag = 'codellama'
        else:
            raise TypeError(f"Unexpect model: {model}.")

        self.llm = LLM(model=self.model,
                       seed=123,
                       tensor_parallel_size=args.gpus,
                       trust_remote_code=True,
                       gpu_memory_utilization=0.9
                       )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_response(self, prompts, max_tokens=1024, temperature=0.01, top_p=0.5):
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                         skip_special_tokens=True, stop=self.tokenizer.eos_token)
        if self.tag in ['mistral']:
            messages_list = [[{"role": "user", "content": p}] for p in prompts]
        else:
            messages_list = [
                [{"role": "system", "content": "You are a helpful SQLite assistant."}, {"role": "user", "content": p}]
                for p in prompts]
        messages_list = self.tokenizer.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=False)
        outputs = self.llm.generate(messages_list, sampling_params)
        return [output.outputs[0].text for output in outputs]


class LLM_Online(object):
    def __init__(self, model="qwen72b", device=[0]):
        None

    def generate_response(self, prompts):
        rs = []
        for prompt in tqdm.tqdm(prompts):
            res = None  # your online LLM
            rs.append(res)
        return rs


def parse_dataset(data_path, mode='dev', dataset='bird'):
    # redirect path
    data_tuples_path = ''
    if dataset == 'bird':
        # data_tuples_path = os.path.join(data_path, dataset, mode, f'{mode}.json')
        data_tuples_path = os.path.join(data_path, dataset, mode, f'{mode}.json')
    elif 'spider_DK' == dataset:
        data_tuples_path = os.path.join(data_path, 'spider', 'Spider_DK.json')
    elif 'spider_real' == dataset:
        data_tuples_path = os.path.join(data_path, 'spider', 'spider-realistic.json')

    elif 'spider_syn' == dataset:
        data_tuples_path = os.path.join(data_path, 'spider', 'dev.json')
    elif 'spider' in dataset:
        if mode == 'test':
            data_tuples_path = os.path.join(data_path, 'spider', 'test.json')
        else:
            data_tuples_path = os.path.join(data_path, 'spider', f'{mode}.json')
    else:
        raise TypeError(f"Unexpect dataset: {dataset}.")

    data_tuples = read_json_file(data_tuples_path)

    return data_tuples


def convert_fk_index(data):
    fk_holder = []
    table_names_original = [i.lower() for i in data['table_names_original']]  # some bug
    column_names_original = [(i[0], i[1].lower()) for i in data['column_names_original']]
    for fk in data["foreign_keys"]:
        tn, col, ref_tn, ref_col = fk[0][0], fk[0][1], fk[1][0], fk[1][1]
        if type(tn) is str:
            tn = tn.lower()
        if type(col) is str:
            col = col.lower()
        if type(ref_tn) is str:
            ref_tn = ref_tn.lower()
        if type(ref_col) is str:
            ref_col = ref_col.lower()
        ref_cid, cid = None, None
        try:
            tid = table_names_original.index(tn)
            ref_tid = table_names_original.index(ref_tn)
            for i, (tab_id, col_org) in enumerate(column_names_original):
                if tab_id == ref_tid and ref_col == col_org:
                    ref_cid = i
                elif tid == tab_id and col == col_org:
                    cid = i
            if ref_cid and cid:
                fk_holder.append([cid, ref_cid])
        except:
            traceback.print_exc()
            print("table_names_original: ", table_names_original)
            print("finding tab name: ", tn, ref_tn)
            print(data)
            # sys.exit()
    return fk_holder


def dump_db_json_schema(db, f):
    '''read table and column info'''

    try:
        conn = sqlite3.connect(db)
    except:
        print(db)
        exit()
    conn.execute('pragma foreign_keys=ON')
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {'db_id': f,
            'table_names_original': [],
            'table_names': [],
            'column_names_original': [(-1, '*')],
            'column_names': [(-1, '*')],
            'column_types': ['text'],
            'primary_keys': [],
            'foreign_keys': []}

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        data['table_names_original'].append(table_name)
        data['table_names'].append(table_name.lower().replace("_", ' '))
        fks = conn.execute("PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
        # print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            data['column_names_original'].append((i, col[1]))
            data['column_names'].append((i, col[1].lower().replace("_", " ")))
            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if 'char' in col_type or col_type == '' or 'text' in col_type or 'var' in col_type:
                data['column_types'].append('text')
            elif 'int' in col_type or 'numeric' in col_type or 'decimal' in col_type or 'number' in col_type \
                    or 'id' in col_type or 'real' in col_type or 'double' in col_type or 'float' in col_type:
                data['column_types'].append('number')
            elif 'date' in col_type or 'time' in col_type or 'year' in col_type:
                data['column_types'].append('time')
            elif 'boolean' in col_type:
                data['column_types'].append('boolean')
            else:
                data['column_types'].append('others')

            if col[5] == 1:
                data['primary_keys'].append(len(data['column_names']) - 1)

    data["foreign_keys"] = fk_holder
    data['foreign_keys'] = convert_fk_index(data)

    return data


def get_schema_dict(db, kk=3):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """
    data = dump_db_json_schema(db, db.split('/')[-1])
    tables = data['table_names_original']
    column_types = data['column_types']
    primary_keys = data['primary_keys']
    foreign_keys = data['foreign_keys']
    column_names = data['column_names_original']

    schema_dict = {
        'tables': {},
        'foreign_keys': []
    }

    for i, table in enumerate(tables):
        t = {}
        for j, c in enumerate(column_names):
            if c[0] == i:
                if j in primary_keys:
                    t[c[1]] = [column_types[j].upper(), True]
                else:
                    t[c[1]] = [column_types[j].upper(), True]
        schema_dict['tables'][table] = t

    for foreign_key in foreign_keys:
        t1 = tables[column_names[foreign_key[0]][0]]
        c1 = column_names[foreign_key[0]][1]
        t2 = tables[column_names[foreign_key[1]][0]]
        c2 = column_names[foreign_key[1]][1]
        schema_dict['foreign_keys'].append([t1, c1, t2, c2])

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    # get exapmles
    for table in schema_dict['tables'].keys():
        try:
            select_query = f'SELECT * FROM `{table}` LIMIT {kk}'
            cursor.execute(select_query)
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info(`{table}`);")
            columns = [column[1] for column in cursor.fetchall()]
            for i, c in enumerate(columns):
                cls_valuse = [f"{row[i][0:100]}..." if type(row[i]) is str and len(row[i]) > 100 else row[i] for row in
                              rows]
                schema_dict['tables'][table][c].append(cls_valuse)
        except Exception as e:
            print(e)
    return schema_dict


def get_example_str(schema_dict, k=1):
    tables = list(schema_dict['tables'].keys())
    examples = {}
    for table in tables:
        table_dict = schema_dict['tables'][table]
        example = []
        for cls in table_dict.keys():
            example.append(table_dict[cls][2])
        example_str = []
        for i, v in enumerate(example[0]):
            example_str.append(tuple([e[i] for e in example]))
            if (i + 1) == k:
                break
        examples[table] = example_str

    e_s = ''
    for key in examples.keys():
        e_s += f"{key}: " + str(examples[key]) + '\n'

    return e_s[:-1]


def get_schmea_str_and_examples(schema_dict):
    schmea_str = ""
    tables = list(schema_dict['tables'].keys())
    examples = {}
    for table in tables:
        if ' ' in table:
            table_str = f'CREATE TABLE "{table}" ('
        else:
            table_str = f"CREATE TABLE {table} ("
        table_dict = schema_dict['tables'][table]

        pk_str = ''
        example = []
        for cls in table_dict.keys():
            try:
                cls_ = f'"{cls}"' if ' ' in cls else cls
                table_str += f"{cls_} {table_dict[cls][0]}, "
                if table_dict[cls][1]:
                    pk_str += cls_ + ', '
                example.append(table_dict[cls][2])
            except Exception as e:
                print(e)
        example_str = []

        try:
            for i, v in enumerate(example[0]):
                example_str.append(tuple([e[i] for e in example]))
        except Exception as e:
            print(e)

        examples[table] = example_str

        if pk_str != '':
            table_str += f"PRIMARY KEY({pk_str[:-2]}), "

        fk_str = ''
        for fk in schema_dict['foreign_keys']:
            if fk[0] == table and fk[2] in tables:
                if fk[3] in schema_dict['tables'][fk[2]].keys():
                    fk = [f'"{f}"' if ' ' in f else f for f in fk]
                    fk_str += f'FOREIGN KEY ({fk[1]}) REFERENCES {fk[2]}({fk[3]}), '
        if fk_str != '':
            table_str += fk_str

        schmea_str += table_str[:-2] + '); '

    schmea_str = schmea_str[:-1]

    e_s = ''
    for key in examples.keys():
        e_s += f"{key}: " + str(examples[key]) + '\n'

    return schmea_str, e_s[:-1]


# parse SQL
def parse_sql_from_string(input_string):
    input_string = input_string.replace('\n', ' ').replace('\t', '')
    rs = ''
    if '```sql' in input_string:
        try:
            sql_pattern = r'```sql(.*?)```'
            all_sqls = []
            for match in re.finditer(sql_pattern, input_string, re.DOTALL):
                all_sqls.append(match.group(1).strip())
            if all_sqls:
                rs = all_sqls[-1]
                if 'SELECT' not in rs and len(all_sqls) > 1:
                    rs = all_sqls[-2]
        except:
            None
    if 'select' in input_string.lower() and rs == '':
        rs = input_string[input_string.find('SELECT'):]
    if ';' in rs:  # end
        rs = rs[:input_string.find(';') + 1]
    if rs == '':
        rs = 'SELECT xx FROM xx'
    return replace_multiple_spaces(rs).replace('```', '')


def replace_multiple_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)


def filter_dict_by_sql(schema_dict, sql):
    schema_dict_ = copy.deepcopy(schema_dict)
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    # tables
    for table in keys:
        if f'from {table.lower()}' not in sql.lower() and f'join {table.lower()}' not in sql.lower():
            schema_dict_['tables'].pop(table, None)
    # columns
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    for table in keys:
        cls_keys = list(schema_dict_['tables'][table].keys())
        cls_keys.sort(key=lambda x: - len(x))
        tabel_dict = copy.deepcopy(schema_dict_['tables'][table])
        for cls in cls_keys:
            if cls.lower() not in sql.lower():
                schema_dict_['tables'][table].pop(cls, None)
        if len(schema_dict_['tables'][table].keys()) == 0:
            # schema_dict_['tables'][table] = tabel_dict  # for COUNT(*)
            for cls in tabel_dict.keys():
                if tabel_dict[cls][1] == True:
                    schema_dict_['tables'][table][cls] = tabel_dict[cls]

        if len(schema_dict_['tables'][table].keys()) == 0:
            schema_dict_['tables'][table][tabel_dict.keys()[0]] = tabel_dict[tabel_dict.keys()[0]]
            schema_dict_['tables'][table][tabel_dict.keys()[1]] = tabel_dict[tabel_dict.keys()[1]]
            # for COUNT(*)

    return schema_dict_


def filter_dict_by_sl(schema_dict, sql):
    schema_dict_ = copy.deepcopy(schema_dict)
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    # tables
    for table in keys:
        if f'{table.lower()}' not in sql.lower():
            schema_dict_['tables'].pop(table, None)
    # columns
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    for table in keys:
        cls_keys = list(schema_dict_['tables'][table].keys())
        cls_keys.sort(key=lambda x: - len(x))
        tabel_dict = copy.deepcopy(schema_dict_['tables'][table])
        for cls in cls_keys:
            if cls.lower() not in sql.lower():
                schema_dict_['tables'][table].pop(cls, None)
        if len(schema_dict_['tables'][table].keys()) == 0:
            # schema_dict_['tables'][table] = tabel_dict  # for COUNT(*)
            for cls in tabel_dict.keys():
                if tabel_dict[cls][1] == True:
                    schema_dict_['tables'][table][cls] = tabel_dict[cls]

        if len(schema_dict_['tables'][table].keys()) == 0:
            schema_dict_['tables'][table][tabel_dict.keys()[0]] = tabel_dict[tabel_dict.keys()[0]]
            schema_dict_['tables'][table][tabel_dict.keys()[1]] = tabel_dict[tabel_dict.keys()[1]]
            # for COUNT(*)

    return schema_dict_


@func_set_timeout(5)
def execute_query_limit(db_path, query):
    error = ''
    result = None
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    cursor = conn.cursor()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result, error


def execute_query(db_path, query):
    try:
        result, error = execute_query_limit(db_path, query)
    except func_timeout.exceptions.FunctionTimedOut:
        error = "SQL execution timeout"
        print("*" * 30, error, query)
        result = None
    except Exception as e:
        error = str(e)
        print("*" * 30, error, query)
        result = None
    return result, error


def replace_syn(data1, data2):
    for i in range(len(data1)):
        if data1[i]['question'] == data2[i]['SpiderQuestion']:
            data1[i]['question'] = data2[i]['SpiderSynQuestion']
    return data1


def eval_all(args):
    dataset = args.dataset
    mode = args.mode
    data_tuples = parse_dataset(args.data_path, mode, dataset)
    batch_size = args.batch_size

    if dataset == 'spider_syn':
        data2 = read_json_file(os.path.join(args.data_path, 'spider', f'dev_syn.json'))
        data_tuples = replace_syn(data_tuples, data2)
        dataset = 'spider'
        args.tag += '_syn'

    if dataset == 'spider_DK':
        args.tag += '_DK'
        dataset = 'spider'

    if dataset == 'spider_real':
        args.tag += '_real'
        dataset = 'spider'

    if dataset == 'bird':
        kk = 5
    else:
        kk = 10
    kkkkk = 1 if dataset == 'bird' else 3

    if 'online' in args.tag:
        generator = LLM_Online()
    else:
        generator = LLM_Model(args.LLM_model)
    tag = args.tag


    # generate SQL
    if True:
        sql_results = []
        data_header = [["NLQ", "Predict", "GOLD", 'database']]
        prompts = []
        for index, row in enumerate(data_tuples):
            if 'spider' in dataset:
                row['SQL'] = row['query']
            if 'drspider' in dataset:
                row['SQL'] = row['query']

            question, db_id = row['question'], row['db_id']
            if dataset == 'spider':
                if mode == 'test':
                    db_path = os.path.join(args.data_path, dataset, 'test_database', db_id, f"{db_id}.sqlite")
                else:
                    db_path = os.path.join(args.data_path, dataset, 'database', db_id, f"{db_id}.sqlite")
            elif dataset == 'drspider':
                db_path = os.path.join(args.data_path, db_id, f"{db_id}.sqlite")
            elif dataset == 'bird':
                db_path = os.path.join(args.data_path, dataset, f'{mode}/{mode}_databases', db_id, f"{db_id}.sqlite")
            else:
                raise TypeError(f"Unexpect dataset: {dataset}.")

            schema_dict = get_schema_dict(db_path, kk=kk)
            database_schema, examples = get_schmea_str_and_examples(schema_dict)
            schema_dict_ = schema_dict

            if dataset == 'bird':
                prompt = [question, schema_dict,
                          f"\n\n/* Question hint */\n{row['evidence']}" if row['evidence'] != '' else '', schema_dict_]
            else:
                prompt = [question, schema_dict, '', schema_dict_]
            prompts.append([database_schema, str(examples), question, row['SQL'], db_id, prompt, db_path])

        n_samples = len(data_tuples)
        n_batches = (n_samples - 1) // batch_size + 1

        prompts_collection = []
        prompts_collection_db = []

        for i in range(n_batches):
            start = i * batch_size
            end = n_samples if i == n_batches - 1 else (i + 1) * batch_size
            batch_prompts = prompts[start: end]
            schema_dicts = []  # only keep the tables

            for j, v in enumerate(batch_prompts):
                batch_prompts[j][1] = get_example_str(batch_prompts[j][5][1], kkkkk)

            # text-to-sql
            final_prompts = [prompt_cw_temp_sft.format(ds=j[0], sr=j[1], qs=j[2], hint=j[5][2], sql='') for j in batch_prompts]
            response_strs = generator.generate_response(prompts=final_prompts)

            def contains_subquery(sql_query, tables):
                sql = sql_query.lower()
                select_num = 0
                join_num = 0
                tmp = sql
                while 'select' in tmp:
                    tmp = tmp[tmp.find('select') + 6:]
                    select_num += 1
                tmp = sql
                while 'join' in tmp:
                    tmp = tmp[tmp.find('select') + 6:]
                    join_num += 1
                table_num = len([key for key in tables if f"from {key.lower()}" in sql or f"join {key.lower()}" in sql])
                if table_num == 1:
                    hard = 1
                elif table_num == 2:
                    hard = 2
                else:
                    hard = 3
                return hard

            nc_idx = []
            continue_sqls = []
            # noisy correction

            for idx, v in enumerate(response_strs):
                pre_sql = parse_sql_from_string(response_strs[idx])
                ex_flg3 = True if execute_query(batch_prompts[idx][6], pre_sql)[1] == '' else False
                hard = contains_subquery(pre_sql, batch_prompts[idx][5][1]['tables'].keys())
                if ex_flg3 == False or hard > 2:
                    common_sql = 'SELECT '
                    continue_sqls.append(common_sql)
                    nc_idx.append(idx)

            # PSG
            if args.PSG:
                cl_prompts = []
                for j, idx in enumerate(nc_idx):
                    v = batch_prompts[idx]
                    ds = get_schmea_str_and_examples(v[5][1])[0]
                    sr = get_example_str(v[5][1], kkkkk)
                    common_sql = continue_sqls[j]
                    if args.eval_sft == 1:
                        cl_prompts.append(
                            prompt_cw_temp_sft.format(ds=ds, sr=sr, qs=v[2], hint=v[5][2], sql=common_sql))
                    else:
                        cl_prompts.append(prompt_cw_temp_sft.format(ds=ds, sr=sr, qs=v[2], hint=v[5][2], sql=common_sql))

                if len(nc_idx) > 0:
                    response_strs_ = generator.generate_response(prompts=cl_prompts)
                    print("%%%%%%%%%%%%%%%%%%", response_strs_[0])
                    for idx, v in enumerate(nc_idx):
                        if execute_query(batch_prompts[v][6], parse_sql_from_string(response_strs_[idx]))[
                            0] is not None:
                            response_strs[v] = response_strs_[idx]

            for j, response_str in enumerate(response_strs):
                database_schema = batch_prompts[j][0]
                question = batch_prompts[j][2]
                gt_sql = replace_multiple_spaces(batch_prompts[j][3])
                if gt_sql.endswith(";;"):
                    gt_sql = gt_sql[:-1]

                if not gt_sql.endswith(";"):
                    gt_sql += ";"

                db_id = batch_prompts[j][4]
                prompt = final_prompts[j]
                print(f"=={start + j + 1}/{len(data_tuples)}=={db_id}=={tag}==================")

                try:
                    if dataset == 'spider':
                        if mode == 'test':
                            db_path = os.path.join(args.data_path, dataset, 'test_database', db_id, f"{db_id}.sqlite")
                        else:
                            db_path = os.path.join(args.data_path, dataset, 'database', db_id, f"{db_id}.sqlite")
                    elif dataset == 'bird':
                        db_path = os.path.join(args.data_path, dataset, f'{mode}/{mode}_databases', db_id,
                                               f"{db_id}.sqlite")
                    else:
                        raise TypeError(f"Unexpect dataset: {dataset}.")

                    SQL_str = parse_sql_from_string(response_str)
                except Exception as e:
                    res = f'error: {str(e)}'
                    print(res, response_str)

                sql_results.append([question, SQL_str, gt_sql, db_id])

                if args.PSG:
                    if j in nc_idx:
                        prompt = prompt_cw_temp_sft.format(ds=batch_prompts[j][0], sr=batch_prompts[j][1], qs=batch_prompts[j][2], hint=batch_prompts[j][5][2], sql=truncate_sql_before_keywords(gt_sql, CLAUSE_KEYWORDS))
                        print(prompt)
                        print(f"Ground: {gt_sql}")
                        # input()

                        prompt_dict = {
                            "input": prompt,
                            "target": gt_sql
                        }

                        prompts_collection.append(prompt_dict)

                else:


                    print(prompt)
                    print(f"Ground: {gt_sql}")

                    prompt_dict1 = {
                        "input": prompt,
                        "db_id": db_id,
                        "target": gt_sql
                    }

                    prompt_dict2 = {
                        "input": prompt,
                        "target": gt_sql
                    }


                    prompts_collection_db.append(prompt_dict1)
                    prompts_collection.append(prompt_dict2)


        if args.PSG:
            filename = os.path.join(args.output_path, f"{tag}_{dataset}_{mode}_{args.flags}_psg.json")

            with open(filename, mode='w',encoding='utf-8') as file:
                json.dump(prompts_collection, file, ensure_ascii=False, indent=4)
        else:
            filename = os.path.join(args.output_path, f"{tag}_{dataset}_{mode}_{args.flags}.json")

            with open(filename, mode='w',encoding='utf-8') as file:
                json.dump(prompts_collection, file, ensure_ascii=False, indent=4)

        if prompts_collection_db:
            filename = os.path.join(args.output_path, f"{tag}_{dataset}_{mode}_db_id_{args.flags}.json")
            with open(filename, mode='w', encoding='utf-8') as file:
                json.dump(prompts_collection_db, file, ensure_ascii=False, indent=4)


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import pynvml

pynvml.nvmlInit()


def usegpu(need_gpu_count=1):
    nouse = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        # 这里的0是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / meminfo.total
        if used < 0.3:
            nouse.append(index)

    if len(nouse) >= need_gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, nouse[:need_gpu_count]))
        # return nouse[:need_gpu_count]
        print(nouse[:need_gpu_count])
        return need_gpu_count
    elif len(nouse) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, nouse))
        return len(nouse)
    else:
        return 0


if __name__ == "__main__":
    from data_process import truncate_sql_before_keywords, CLAUSE_KEYWORDS
    parser = argparse.ArgumentParser(description='SQL')
    parser.add_argument("--dataset", default='bird', type=str)
    parser.add_argument("--data_path", default='/data/vda/dataset', type=str)
    parser.add_argument("--output_path", default='/data/vda/dataset', type=str)
    parser.add_argument("--mode", default='dev', type=str)
    parser.add_argument("--PSG", action='store_true', default=False)
    parser.add_argument("--tag", default='SQL-o1', type=str)
    parser.add_argument("--gpus", default=4, type=int)
    parser.add_argument("--eval_sft", default=1, type=int)
    parser.add_argument("--flags", default='0', type=str)
    # parser.add_argument("--LLM_model", default='meta-llama/Llama-3-8B-Instruct', type=str)
    parser.add_argument("--LLM_model", default='/data/vda/saves/llama3-8b', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()
    usegpu(need_gpu_count=args.gpus)
    print(args)
    eval_all(args)
