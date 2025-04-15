TEMPLATE = "-- Database {}:\n{}\n-- Question: {}\n -- SQL:\n"
TEMPLATE_CORR = "-- Database {}:\n{}\n-- Question: {}\n -- Buggy SQL:\n{}\n -- Fixed SQL:\n"

TEMPLATE_EVAL = "-- Utterance: {}\n-- SQL:\n{}"
TEMPLATE_EVAL_RES = "-- Utterance: {}\n-- SQL:\n{}\n-- Result:\n{}"

INST_CODELLAMA_GEN = "[INST] Given database schema and a question in natural language, generate the corresponding SQL query.\n\n{} [/INST]"

INST_CODELLAMA_ITER_CORR = "[INST] Given database schema and a question in natural language, correct the buggy SQL query and generate a fixed SQL query.\n\n{} [/INST]"

INST_CODELLAMA_EVAL = "[INST] Answer the following Yes/No question: Is the SQL correct given the utterance?\n\n{}\n-- Answer: [/INST]" 

INST_CODELLAMA_EVAL_DB = "[INST] Answer the following Yes/No question: Is the SQL correct given the database schema and the utterance?\n\n{}\n-- Answer: [/INST]"

INST_CODELLAMA_EVAL_RES = "[INST] Answer the following Yes/No question: Is the SQL correct given the utterance and its result?\n\n{}\n-- Answer: [/INST]"

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

TEMPLATE_GSM8K = "[INST] ## Given questions in the comment, use python programs to produce the correct answers with the 'answer' variable.\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_tylenol_per_tablet = 375\nmg_tylenol_taken_each_time = 2 * mg_tylenol_per_tablet\nhours_per_day = 24\ntimes_per_day = hours_per_day / 6\nmg_each_day = mg_tylenol_taken_each_time * times_per_day\nanswer = mg_each_day\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\nn_easter_eggs = 63\nunit_times = 2\ntotal_units = unit_times + 1\nn_easter_eggs_per_unit = n_easter_eggs / total_units\nn_easter_eggs_helen = n_easter_eggs_per_unit * 1\nn_easter_eggs_hannah = n_easter_eggs_per_unit * 2\nanswer = n_easter_eggs_hannah\n\n## {}\n [/INST] ## Python Program:\n"

TEMPLATE_GSM8K_EVAL = "[INST] ## Answer the following Yes/No question: Is the python program correct given the problem in the comment?\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_tylenol_per_tablet = 375\nmg_tylenol_taken_each_time = 2 * mg_tylenol_per_tablet\nhours_per_day = 24\ntimes_per_day = hours_per_day / 6\nmg_each_day = mg_tylenol_taken_each_time * times_per_day\nanswer = mg_each_day\n## Answer: Yes\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_per_tablet = 375\nn_tablets_per_day = 2\nn_tablets_per_6hrs = n_tablets_per_day / 6\nmg_per_6hrs = mg_per_tablet * n_tablets_per_6hrs\nanswer = mg_per_6hrs\n## Answer: No\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\nn_easter_eggs = 63\nunit_times = 2\ntotal_units = unit_times + 1\nn_easter_eggs_per_unit = n_easter_eggs / total_units\nn_easter_eggs_helen = n_easter_eggs_per_unit * 1\nn_easter_eggs_hannah = n_easter_eggs_per_unit * 2\nanswer = n_easter_eggs_hannah\n## Answer: Yes\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\neggs_in_yard = 63\neggs_found_by_hannah = 2 * eggs_in_yard\neggs_found_by_helen = eggs_found_by_hannah / 2\nanswer = eggs_found_by_hannah\n## Answer: No\n\n## {}\n## Python Program:\n{}\n## Answer: [/INST]"

TEMPLATE_GSM8K_EVAL_RES = "[INST] ## Answer the following Yes/No question: Is the python program correct given its result and the problem in the comment?\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_tylenol_per_tablet = 375\nmg_tylenol_taken_each_time = 2 * mg_tylenol_per_tablet\nhours_per_day = 24\ntimes_per_day = hours_per_day / 6\nmg_each_day = mg_tylenol_taken_each_time * times_per_day\nanswer = mg_each_day\n## Result: 3000.0\n## Answer: Yes\n\n## James takes 2 Tylenol tablets that are 375 mg each, every 6 hours. How many mg does he take a day?\n## Python Program:\nmg_per_tablet = 375\nn_tablets_per_day = 2\nn_tablets_per_6hrs = n_tablets_per_day / 6\nmg_per_6hrs = mg_per_tablet * n_tablets_per_6hrs\nanswer = mg_per_6hrs\n## Result: 125.0\n## Answer: No\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\nn_easter_eggs = 63\nunit_times = 2\ntotal_units = unit_times + 1\nn_easter_eggs_per_unit = n_easter_eggs / total_units\nn_easter_eggs_helen = n_easter_eggs_per_unit * 1\nn_easter_eggs_hannah = n_easter_eggs_per_unit * 2\nanswer = n_easter_eggs_hannah\n## Result: 42\n## Answer: Yes\n\n## There were 63 Easter eggs in the yard. Hannah found twice as many as Helen. How many Easter eggs did Hannah find?\n## Python Program:\neggs_in_yard = 63\neggs_found_by_hannah = 2 * eggs_in_yard\neggs_found_by_helen = eggs_found_by_hannah / 2\nanswer = eggs_found_by_hannah\n## Result: 126\n## Answer: No\n\n## {}\n## Python Program:\n{}\n## Result: {}\n## Answer: [/INST]"

GSM8K_ITER_CORR = "[INST] ## Given the question in the comment, correct the buggy python program and generate a fixed python program to produce the correct answer with the 'answer' variable.\n\n## {}\n## Buggy Python Program:\n{}\n [/INST] ## Fixed Python Program:\n"