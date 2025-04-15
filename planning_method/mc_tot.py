from utils.constants import TEMPLATE, SQL_KEYWORDS, INST_CODELLAMA_GEN
from utils.inference_utils import segment_step
from utils.normalize_sql import normalize_sql

from copy import deepcopy
from func_timeout import func_timeout
from transformers import GenerationConfig

import heapq
import numpy as np
import json
import torch


def mc_tot(example, generator, evaluator, retriever_gen, retriever_eval, args, log):
    config = json.load(open(args.generation_config))
    evaluation_config = json.load(open(args.evaluation_config))

    if retriever_gen is None:
        model_inp = TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    else:
        demos = retriever_gen.retrieve(example["question"])
        model_inp = "\n\n".join([TEMPLATE.format(ex["db_id"], ex["schema"], ex["question"]) + ex["sql"] for ex in demos])
        model_inp = model_inp + "\n\n" + TEMPLATE.format(example["db_id"], example["schema"], example["question"])
    
    prompt = model_inp
    if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
        prompt = INST_CODELLAMA_GEN.format(prompt) + " SELECT"
    
    partial_sql = ""
    current_score = 18
    heap = []
    candidates = []
    sql_score_cache = {}

    for t in range(50):
        # Step 1: Prompt the generator and sample initial steps.
        responses = generator.generate(prompt, config)
        if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
            sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in responses if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
        else:
            sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in responses if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))

        # Segment generator completions to find the first new step.
        steps = set([
            (
                segment_step(sql[len(partial_sql):].lstrip()).rstrip()
                if len(sql) > len(partial_sql)
                else sql
            )
            for sql in sql_completions
        ])

        # Step 2: For each new step, score it with the best the Monte-Carlo rollout.
        step_score = {}
        for s in steps:
            # Sample Monte-Carlo rollouts.
            mc_prompt = model_inp + partial_sql
            if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
                mc_prompt = INST_CODELLAMA_GEN.format(model_inp) + " " + partial_sql + " " + s

            mc_rollouts = generator.generate(mc_prompt, config)
            if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
                mc_sql_completions = list(set([normalize_sql(r.split(" [/INST] ")[-1].split("\n\n")[0]) for r in mc_rollouts if r.split(" [/INST] ")[-1].split("\n\n")[0] != ""]))
            else:
                mc_sql_completions = list(set([normalize_sql(r.split("-- SQL:\n")[-1].split("\n\n")[0]) for r in mc_rollouts if r.split("-- SQL:\n")[-1].split("\n\n")[0] != ""]))

            # Evaluate Monte-Carlo rollouts.
            if args.evaluator_name == "oracle":
                try:
                    scores = func_timeout(300.0, evaluator.score, args=(example["db_id"], example["question"], mc_sql_completions, example["sql"]))
                except:
                    scores = [0 for c in mc_sql_completions]
            elif retriever_eval is None:
                try:
                    scores = func_timeout(300.0, evaluator.score, args=(example["db_id"], example["question"], mc_sql_completions, evaluation_config))
                except:
                    scores = [0 for c in mc_sql_completions]
            else:
                try:
                    scores = func_timeout(300.0, evaluator.score_fewshot, args=(example["db_id"], example["question"], mc_sql_completions, retriever_eval, evaluation_config))
                except:
                    scores = [0 for c in mc_sql_completions]

            # Find the plan with highest score. Scores are negated for min heap implementation.
            step_score[s] = min(scores)

        # Step 3: Update the heap memory with new steps and scores.
        for k, v in step_score.items():
            if v <= 0: # prune if all rollouts have execution errors
                heapq.heappush(heap, (v, partial_sql + " " + k))

        # If the heap is empty before finding the first complete program (e.g. all of them are non-executable), return the first initial plan.
        if len(heap) == 0:
            if not partial_sql.endswith(";"):
                partial_sql = sql_completions[0]
            break

        # Step 4: Pop out the current best (partial) plan in heap. If complete, return it.
        current_score, partial_sql = heapq.heappop(heap)
        if partial_sql.endswith(";"):
            break
        else:
            partial_sql = normalize_sql(partial_sql).rstrip(";")

            if generator.base_model_name.startswith("codellama") and "Instruct" in generator.base_model_name:
                prompt = INST_CODELLAMA_GEN.format(model_inp) + " " + partial_sql
            else:
                prompt = model_inp + partial_sql
    
        # For debugging
        # print(current_score, partial_sql)
        # print(heap)

    answer = partial_sql.replace("\n", " ")
    heapq.heappush(heap, (current_score, partial_sql))

    example_log = deepcopy(example)
    example_log["heap"] = dict([(t[1], t[0]) for t in heap])
    log.append(example_log)

    return answer