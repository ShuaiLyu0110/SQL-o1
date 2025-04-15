from accelerate.utils import set_seed
from tqdm import tqdm

import argparse
import json
import numpy as np
import openai
import random
import torch


def select_models(args):
    generator = None
    evaluator = None
    retriever_gen = None
    retriever_eval = None

    # if args.generator_name.startswith("gpt"):
    #     from generators.openai_generator import OpenaiGenerator
    #     generator = OpenaiGenerator(args.generator_name)
    # else:

    if args.generator_peft_dir == "":
        from generators.hf_generator import HFGenerator
        generator = HFGenerator(args.generator_name, device="cuda")
    else:
        from generators.hf_generator import HFLoraGenerator
        generator = HFLoraGenerator(args.generator_name, args.generator_peft_dir, device="cuda")

    evaluator_name = args.evaluator_name.split('/')[1]
    print(evaluator_name)

    if evaluator_name == "oracle":
        from evaluators.oracle_evaluator import OracleEvaluator
        evaluator = OracleEvaluator(args.db_path, args.oracle_prob)
    elif evaluator_name.startswith("codellama"):
        if args.evaluator_peft_dir == "":
            from evaluators.codellama_evaluator import CodeLlamaEvaluator
            evaluator = CodeLlamaEvaluator(args.evaluator_name, args.db_path, device="cuda")
        else:
            from evaluators.codellama_evaluator import CodeLlamaLoraEvaluator
            evaluator = CodeLlamaLoraEvaluator(args.evaluator_name, args.evaluator_peft_dir, args.db_path, device="cuda")
    elif evaluator_name.startswith("gpt"):
        from evaluators.openai_evaluator import OpenaiEvaluator
        evaluator = OpenaiEvaluator(args.evaluator_name, args.db_path)
    # elif "starcoder" in args.evaluator_name:
    #     from evaluators.starcoder_evaluator import StarcoderEvaluator
    #     evaluator = StarcoderEvaluator(args.evaluator_name, args.db_path, device="cuda")

    if args.retriever_name == "bm25":
        from retrievers.bm25 import BM25Retriever
        if args.retriever_corpus_gen:
            retriever_gen = BM25Retriever(args.retriever_name, args.retriever_corpus_gen, args.retrieve_k)
        if args.retriever_corpus_eval:
            retriever_eval = BM25Retriever(args.retriever_name, args.retriever_corpus_eval, args.retrieve_k)

    return generator, evaluator, retriever_gen, retriever_eval


def select_method(method_name):
    if method_name == "mctot":
        from planning_methods.mc_tot import mc_tot
        return mc_tot
    elif method_name == "greedy":
        from planning_methods.greedy import greedy
        return greedy
    elif method_name == "rerank":
        from planning_methods.rerank import rerank
        return rerank
    elif method_name == "iter_corr":
        from planning_methods.iter_correction import iter_correction
        return iter_correction
    else:
        raise Exception("Invalid method.")


def inference(generator, evaluator, retriever_gen, retriever_eval, args):
    test_data = json.load(open(args.test_fname))
    method = select_method(args.method_name)

    results = []
    log = []

    for ex in tqdm(test_data):
        res_sql = method(ex, generator, evaluator, retriever_gen, retriever_eval, args, log)
        if args.dataset_name == "spider":
            results.append(res_sql + "\t" + ex["db_id"])  # spider
        elif args.dataset_name == "bird":
            results.append(res_sql + "\t----- bird -----\t" + ex["db_id"])
        else:
            raise Exception("Invalid dataset name.")
        print(ex)
        print(res_sql)


    if args.dataset_name == "spider":
        out = open(args.result_fname, "w+", encoding="utf-8")
        out.write("\n".join(results))
        out.close()
    elif args.dataset_name == "bird":
        out = open(args.result_fname, "w+", encoding="utf-8")
        json.dump(results, out, indent=2)
        out.close()
    else:
        raise Exception("Invalid dataset name.")

    if args.log_fname != "":
        out = open("log/" + args.log_fname, "w+", encoding="utf-8")
        json.dump(log, out, indent=2)
        out.close()


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--test_fname', type=str, default='data/spider_dev.json')
    args_parser.add_argument('--result_fname', type=str, default='spider_dev.sql')
    args_parser.add_argument('--log_fname', type=str, default='spider_dev.json')
    args_parser.add_argument('--dataset_name', type=str, default='spider')
    args_parser.add_argument('--db_path', type=str, default='../spider/database')

    args_parser.add_argument('--method_name', type=str, default='mctot')
    args_parser.add_argument('--generator_name', type=str, default='bigcode/starcoderbase')
    args_parser.add_argument('--generator_peft_dir', type=str, default='')
    args_parser.add_argument('--evaluator_name', type=str, default='') #codellama/CodeLlama-13b-Instruct-hf
    args_parser.add_argument('--evaluator_peft_dir', type=str, default='')
    args_parser.add_argument('--oracle_prob', type=float, default=1.0)
    args_parser.add_argument('--retriever_name', type=str, default='')
    args_parser.add_argument('--retriever_corpus_gen', type=str, default='data/spider_train.json')
    args_parser.add_argument('--retriever_corpus_eval', type=str, default='')
    args_parser.add_argument('--retrieve_k', type=int, default=1)

    args_parser.add_argument('--seed', type=int, default=42)
    args_parser.add_argument('--api_key', type=str, default='')
    args_parser.add_argument('--generation_config', type=str, default='generation_configs/temp_sampling.json')
    args_parser.add_argument('--evaluation_config', type=str, default='')

    args = args_parser.parse_args()
    set_seed_all(args.seed)

    if args.api_key:
        openai.api_key = args.api_key

    generator, evaluator, retriever_gen, retriever_eval = select_models(args)

    inference(generator, evaluator, retriever_gen, retriever_eval, args)