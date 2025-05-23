# SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL

## Overview


![](./SQL-o1.png)

## Introduction

PyTorch implementation for SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL.

## Dependence
```bash
conda create -n moeuie python=3.11
conda activate SQL-o1
pip install torch==2.3.0
pip install -r requirements.txt
```

## Data Preparation (Schema-Aware Data + PSG)
### 1.1 Please place the downloaded dataset files in the directory structure as shown below.[README.md](..%2F..%2FWechat_Files%2FWeChat%20Files%2Fwxid_wfdsrq1u3sxe22%2FFileStorage%2FFile%2F2025-04%2FREADME.md)
```bash
SQL-o1/
└──dataset/
    ├── spider/                  
        ├── train.json
        ├── tables.json
        ├── Spider_DK.json
        ├── spider-realistic.json
        ├── dev_syn.json
        ├── ...
        ├── dev.json
        ├── test.json
        ├── test_database/
        └── database/ 
    ├── bird/                 
        ├── train/
            ├── train.json
            ├── train_tables.json
            ├── ...
            └── train_databases/
        ├── dev/                    
            ├── dev.json                   
            ├── dev_tables.json
            ├── ...    
            └── dev_databases/  
                                     
```

### 1.2 Run the script below and replace the parameters with your actual values.
```bash
python preprocess_data.py --dataset spider|spider_real|spider_DK|spider_syn --mode train --LLM_model  meta-llama/Meta-Llama-3-8B-Instruct --PSG --data_path /data/vda/dataset --output_path ./dataset 
python preprocess_data.py --dataset bird --mode train --LLM_model meta-llama/Meta-Llama-3-8B-Instruct --PSG --data_path /data/vda/dataset --output_path ./dataset 
```
## SFT for Model
####  Train the model based on the data obtained from the previous step using LlamaFactory.
```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```
```bash[kill_llm_api.sh](kill_llm_api.sh)
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export  --model_name_or_path /home/huggingface/meta-llama/Llama-3-8B-Instruct --adapter_name_or_path /data/vda/saves/llama3-8b/sft/lora  --template llama3 --finetuning_type lora --use_dora --export_dir /data/vda/llama3_merge/ --export_size 2 --export_legacy_format False
```

## MCTS Search for Model


### 2.1 Prepare the test data by ensuring it is properly formatted
```bash
python preprocess_data.py --dataset spider|spider_real|spider_DK|spider_syn --mode dev(test: spider_test) --LLM_model  meta-llama/Meta-Llama-3-8B-Instruct  --data_path /data/vda/dataset --output_path ./dataset 
python preprocess_data.py --dataset bird --mode dev --LLM_model meta-llama/Meta-Llama-3-8B-Instruct  --data_path /data/vda/dataset --output_path ./dataset 
```
### 2.2 Start LLM API for Models
```bash
CUDA_VISIBLE_DEVICES=0 API_PORT=8000 nohup python src/llm_api.py --model_name_or_path  /data/vda/llama3_merge/  --template llama3 --temperature 0.9 >> result_llm_api_0.log 2>&1 &
```
### 2.3 MCTS Explore for Model (Results collection & Please replace it with your own valid parameters. )
```bash
nohup python _run_explore.py --task_name bird >> result_mcts_0.txt 2>&1 &
python validation_results.py --json_path ./mcts_results/bird_mcts_dev.json ( | spider_mcts_dev.json | spider_syn.json | spider_DK.json | spider_real.json | spider_test.json ) --db_root_path ./dataset/bird/dev/dev_databases --num_cpus 1 --diff_json_path ./dataset/bird/dev/dev.json  --output_file  spider_dev.sql (...)
```

### 2.4 Close API of Model & Test the quality of the generated .sql file.
```bash
bash kill_llm_api.sh
```

## BibTex
If this work contributes to your research, please cite it as follows:
```bash
@misc{lyu2025sqlo1selfrewardheuristicdynamic,
      title={SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL}, 
      author={Shuai Lyu and Haoran Luo and Zhonghong Ou and Yifan Zhu and Xiaoran Shang and Yang Qin and Meina Song},
      year={2025},
      eprint={2502.11741},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2502.11741}, 
}
```
For further questions, please contact: [Lxb_savior@bupt.edu.cn](Lxb_savior@bupt.edu.cn)
## Acknowledgement
This repository builds upon [LLM-Reasoners](https://github.com/maitrix-org/llm-reasoners) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We appreciate their excellent work.
