import subprocess
import sys 

mode = sys.argv[1] # "local", "finetune", "inference", "inference_wiki_stance", "inference_wiki_sentiment"

# main datasets:
datasets = ["cstance", "pstance2", "semeval"]
targets_ls = [
    ["face_masks", "fauci", "school_closures", "stay_at_home_orders"],
    ["bernie", "trump", "biden"],
    ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"]
]
models = ["llama-3.2-3b-instruct", "llama-3.2-3b-instruct-web", 
            "llama-3.1-8b-instruct", "llama-3.1-8b-instruct-cot", "llama-3.1-8b-instruct-cot-fewshot",
            "llama-3.1-8b-instruct-web", "llama-3.1-8b-instruct-web-cot", "llama-3.1-8b-instruct-web-cot-fewshot",
            "qwen2.5-3b-instruct", "qwen2.5-3b-instruct-web",
            "qwen2.5-7b-instruct", "qwen2.5-7b-instruct-cot", "qwen2.5-7b-instruct-cot-fewshot",
            "qwen2.5-7b-instruct-web", "qwen2.5-7b-instruct-web-cot", "qwen2.5-7b-instruct-web-cot-fewshot",
            "qwen2.5-14b-instruct", "qwen2.5-14b-instruct-web",
            "qwen2.5-32b-instruct", "qwen2.5-32b-instruct-web",
            "gpt-4o-mini", "gpt-4o-mini-cot", "gpt-4o-mini-cot-fewshot",
            "gpt-4o-mini-web", "gpt-4o-mini-web-cot", "gpt-4o-mini-web-cot-fewshot",
            "claude-3-haiku", "claude-3-haiku-cot", "claude-3-haiku-cot-fewshot", 
            "claude-3-haiku-web", "claude-3-haiku-web-cot", "claude-3-haiku-web-cot-fewshot"]

# LOCAL INFERENCE WITHOUT FINE-TUNING
# supported models: 
# llama-3.2-3b-instruct, llama-3.2-3b-instruct-web,
# llama-3.1-8b-instruct, llama-3.1-8b-instruct-cot, llama-3.1-8b-instruct-cot-fewshot,
# llama-3.1-8b-instruct-web, llama-3.1-8b-instruct-web-cot, llama-3.1-8b-instruct-web-cot-fewshot,
# qwen2.5-3b-instruct, qwen2.5-3b-instruct-web,
# qwen2.5-7b-instruct, qwen2.5-7b-instruct-cot, qwen2.5-7b-instruct-cot-fewshot,
# qwen2.5-7b-instruct-web, qwen2.5-7b-instruct-web-cot, qwen2.5-7b-instruct-web-cot-fewshot,
# qwen2.5-14b-instruct, qwen2.5-14b-instruct-web,
# qwen2.5-32b-instruct, qwen2.5-32b-instruct-web
if mode == "local":
    device = 2
    dataset = "semeval"
    target = "legalization_of_abortion"
    model = "llama-3.1-8b-instruct-cot-fewshot"
    stance = "default"
    command = f"CUDA_VISIBLE_DEVICES={device} && python scripts/local.py --mode local --dataset {dataset} --target {target} --model {model} --stance {stance}"
    tmux_name = f"[cuda_{device}]_local_{dataset}_{target}_{model}_{stance}"
    subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, command], check=True)
    print(tmux_name)

# LOCAL FINE-TUNING
# supported models: llama-3.1-8b-instruct, qwen2.5-7b-instruct
# llama-3.1-8b-instruct-web, qwen2.5-7b-instruct-web
if mode == "finetune":
    device = 1
    dataset = "cstance"
    target = "face_masks"
    model = "llama-3.1-8b-instruct"
    epoch = 1
    rank = 32
    command = f"CUDA_VISIBLE_DEVICES={device} && python scripts/local.py --mode finetune --dataset {dataset} --target {target} --model {model} --epoch {epoch} --rank {rank}"
    tmux_name = f"[cuda_{device}]_finetune_{dataset}_{target}_{model}_{epoch}_r{rank}"
    subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, command], check=True)
    print(tmux_name)

# LOCAL INFERENCE AFTER FINE-TUNING
# supported models: llama-3.1-8b-instruct, qwen2.5-7b-instruct
# llama-3.1-8b-instruct-web, qwen2.5-7b-instruct-web
if mode == "inference":
    device = 1
    dataset = "cstance"
    model = "llama-3.1-8b-instruct"
    epoch = 1
    rank = 32
    command = f"CUDA_VISIBLE_DEVICES={device} && python scripts/local.py --mode inference --dataset {dataset} --model {model} --epoch {epoch} --rank {rank}"
    tmux_name = f"[cuda_{device}]_inference_{dataset}_{model}_{epoch}_r{rank}"
    subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, command], check=True)
    print(tmux_name)


# LOCAL INFERENCE ON WIKI
# supported models: llama-3.1-8b-instruct, llama-3.1-8b-instruct-web, 
# qwen2.5-7b-instruct, qwen2.5-7b-instruct-web
if mode == "inference_wiki_stance":
    device = 1
    model = "llama-3.1-8b-instruct"
    stance = "default"
    command = f"export CUDA_VISIBLE_DEVICES={device} && python scripts/local.py --mode inference_wiki_stance --model {model} --stance {stance}"
    tmux_name = f"[cuda_{device}]_inference_wiki_stance_{model}_{stance}"
    subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, command], check=True)
    print(tmux_name)
# supported models: llama-3.1-8b-instruct, llama-3.1-8b-instruct-web,
# qwen2.5-7b-instruct, qwen2.5-7b-instruct-web
if mode == "inference_wiki_sentiment":
    device = 1
    model = "llama-3.1-8b-instruct"
    stance = "default"
    command = f"export CUDA_VISIBLE_DEVICES={device} && python scripts/local.py --mode inference_wiki_sentiment --model {model} --stance {stance}"
    tmux_name = f"[cuda_{device}]_inference_wiki_sentiment_{model}_{stance}"
    subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, command], check=True)
    print(tmux_name)
