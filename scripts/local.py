from utils import get_hf_data, get_hf_wiki_data, parse_llama_outputs, parse_qwen_outputs
from config import prefix

from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from vllm import LLM, SamplingParams
import wandb

import os
import time
import re
from tqdm import tqdm
import pandas as pd


def get_model_name_arg(model_name):
    if model_name.startswith("llama-3.2-3b-instruct"):
        model_name_arg = "unsloth/Llama-3.2-3B-Instruct"
    elif model_name.startswith("llama-3.2-1b-instruct"):
        model_name_arg = "unsloth/Llama-3.2-1B-Instruct"
    elif model_name.startswith("llama-3.1-8b-instruct"):
        model_name_arg = "unsloth/Meta-Llama-3.1-8B-Instruct"
    elif model_name.startswith("qwen2.5-1.5b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-1.5B-Instruct"
    elif model_name.startswith("qwen2.5-3b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-3B-Instruct"
    elif model_name.startswith("qwen2.5-7b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-7B-Instruct"
    elif model_name.startswith("qwen2.5-14b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-14B-Instruct"
    elif model_name.startswith("qwen2.5-32b-instruct"):
        model_name_arg = f"unsloth/Qwen2.5-32B-Instruct"
    return model_name_arg


def finetune(dataset, target, model_name="llama-3.2-1b-instruct", rank=32, epoch=1):
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # load_in_4bit = False

    if rank==16:
        rank_postfix = ""
    else:
        rank_postfix = f"_r{rank}"

    if load_in_4bit:
        quant_status = ""
    else:
        quant_status = "-unquantized"

    if epoch == 1:
        model_name_arg = get_model_name_arg(model_name)
    else:
        model_name_arg = f"{prefix}epoch-{epoch-1}/{model_name}{quant_status}/lora_{dataset}_{target}{rank_postfix}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_arg, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        fix_tokenizer=False,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if model_name.startswith("llama"):
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    elif model_name.startswith("qwen"):
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "chatml",
        )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    data = get_hf_data(dataset, "train", target, "default", 
                       cot="cot" in model_name, 
                       use_web="web" in model_name)
    data = data.map(formatting_prompts_func, batched = True,)

    run = wandb.init(
        project = "stance",
        name = f"{model_name}{quant_status} {dataset}_{target} (from epoch {epoch-1} ckp)",
    )

    if not os.path.exists(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/{dataset}_{target}{rank_postfix}"):
        os.makedirs(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/{dataset}_{target}{rank_postfix}")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = data,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 0,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 100,
            learning_rate = 1e-4, # 2e-4
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            # lr_scheduler_type = "linear",
            lr_scheduler_type = "constant",
            seed = 3406+epoch,
            output_dir = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/outputs/{dataset}_{target}{rank_postfix}",
            report_to = "wandb", # Use this for WandB etc
        ),
    )
    if model_name.startswith("llama"):
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif model_name.startswith("qwen"):
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "user\n",
            response_part = "assistant\n",
        )
    trainer.train()
    run.finish()
    
    if not os.path.exists(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/"):
        os.makedirs(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/")
    model.save_pretrained(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_{dataset}_{target}{rank_postfix}")
    tokenizer.save_pretrained(f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_{dataset}_{target}{rank_postfix}")


def inference(dataset, target, modified_stance, model_name="llama-3.2-1b-instruct", rank=16, epoch=0, use_vllm=True):

    if rank==16:
        rank_postfix = ""
    else:
        rank_postfix = f"_r{rank}"

    # 5120 due to long wiki/web info repetition in fewshot
    max_seq_length = 5120 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # load_in_4bit = False

    if load_in_4bit:
        quant_status = ""
    else:
        quant_status = "-unquantized"

    if epoch != 0:
        model_name_arg = f"{prefix}epoch-{epoch}/{model_name}{quant_status}/lora_{dataset}_{target}{rank_postfix}"
    else:
        model_name_arg = get_model_name_arg(model_name)

    if use_vllm and epoch == 0:
        llm = LLM(model=(model_name_arg + "-bnb-4bit").lower(), max_model_len=max_seq_length,
                  device="cuda")
    elif use_vllm and epoch != 0:
        llm = LLM(model=(model_name_arg).lower(), max_model_len=max_seq_length,
                  device="cuda")
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_arg, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            fix_tokenizer=False
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    cot = "cot" in model_name
    data = get_hf_data(dataset, "test", target, modified_stance, 
                    cot=cot, 
                    use_web="web" in model_name,
                    fewshot="fewshot" in model_name)
    predictions = []
    raw_predictions = []
    ground_truth = data["ground_truth"]
    answers_ls = []
    if "cot" in model_name:
        chains_of_thought = []
        # if "fewshot" not in model_name:
            # conclusions = []
    
    n_candidates = 1
    if use_vllm:
        sampling_params = SamplingParams(temperature=0, max_tokens=1024)
        outputs = llm.chat(data["conversations"], sampling_params)
        answers_ls = [[output.outputs[0].text] for output in outputs]
    else:
        for i in tqdm(range(len(data))):
            # if i > 0: # for debugging
                # break
            inputs = tokenizer.apply_chat_template(
                data[i]["conversations"], 
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")
            temperature = 0.001
            outputs = model.generate(input_ids = inputs, max_new_tokens = 1024,
                        use_cache = True, temperature = temperature,
                        num_return_sequences=n_candidates)
            
            if model_name.startswith("llama"):
                llama_outputs = tokenizer.batch_decode(outputs)
                answers = [parse_llama_outputs(output, cot=cot) for output in llama_outputs]
            elif model_name.startswith("qwen"):
                qwen_outputs = tokenizer.batch_decode(outputs)
                answers = [parse_qwen_outputs(output, cot=cot) for output in qwen_outputs]
            answers_ls.append(answers)
    
    for answers in answers_ls:
        
        if "cot" in model_name:
            chain_of_thought = answers[0]
            chains_of_thought.append(chain_of_thought)
            # if "fewshot" in model_name:
            matches = re.findall(r'\b(FAVOR|AGAINST|NONE)\b', chain_of_thought)
            ans = matches[-1] if matches else "INVALID"
            sample_predictions = [ans]
            predictions.append(sample_predictions)
        else:
            raw_predictions.append(answers)
            sample_predictions = []
            # eliminate non-alphabetical characters
            answers = ["".join(filter(str.isalpha, ans)) for ans in answers]
            for ans in answers:
                if ans.lower() in ["favor", "favour", "favorable", "favourable"]:
                    sample_predictions.append("FAVOR")
                elif ans.lower() == "against":
                    sample_predictions.append("AGAINST")
                elif ans.lower() in ["none", "neutral"] and dataset != "pstance2":
                    sample_predictions.append("NONE")
                else:
                    sample_predictions.append("INVALID")
            predictions.append(sample_predictions)

    # Answer cleansing a la zero-shot CoT (Kojima et al. 2022)
    if "cot" in model_name:
        print("Answer Extraction!")
        conclusions_col = []
        parsed_predictions = []
        aug_data = get_hf_data(dataset, "test", target, modified_stance,
                            cot=cot, 
                            use_web="web" in model_name,
                            chains_of_thought=chains_of_thought,
                            fewshot="fewshot" in model_name)
        if use_vllm:
            outputs = llm.chat(aug_data["conversations"], sampling_params)
            conclusions_ls = [[output.outputs[0].text] for output in outputs]
        else:
            for i in tqdm(range(len(data))):
                # if i > 0: # for debugging
                    # break
                inputs = tokenizer.apply_chat_template(
                    aug_data[i]["conversations"], 
                    tokenize = True,
                    add_generation_prompt = True, # Must add for generation
                    return_tensors = "pt",
                ).to("cuda")
            
                outputs = model.generate(input_ids = inputs, max_new_tokens = 1024,
                            use_cache = True, temperature = temperature,
                            num_return_sequences=n_candidates)
                
                if model_name.startswith("llama"):
                    llama_outputs = tokenizer.batch_decode(outputs)
                    answers = [parse_llama_outputs(output, cot=cot) for output in llama_outputs]
                elif model_name.startswith("qwen"):
                    qwen_outputs = tokenizer.batch_decode(outputs)
                    answers = [parse_qwen_outputs(output, cot=cot) for output in qwen_outputs]
                conclusions = answers[0]
                conclusions_ls.append(conclusions)

        for conclusions in conclusions_ls:
            conclusion = conclusions[0]
            conclusions_col.append(conclusion)
            matches = re.findall(r'\b(FAVOR|FAVOUR|Favor|Favour|favor|favour|AGAINST|Against|against|NONE|NEUTRAL|None|Neutral|none|neutral)\b', conclusion)
            ans = matches[0] if matches else "INVALID"
            answers = [ans]
            sample_predictions = []
            # eliminate non-alphabetical characters
            answers = ["".join(filter(str.isalpha, ans)) for ans in answers]
            for ans in answers:
                if ans.lower() in ["favor", "favour", "favorable", "favourable"]:
                    sample_predictions.append("FAVOR")
                elif ans.lower() == "against":
                    sample_predictions.append("AGAINST")
                elif ans.lower() in ["none", "neutral"] and dataset != "pstance2":
                    sample_predictions.append("NONE")
                else:
                    sample_predictions.append("INVALID")
            parsed_predictions.append(sample_predictions)

    results_df = {"ground_truth": ground_truth}
    for i in range(n_candidates):
        results_df[f"prediction_{i}"] = [pred[i] for pred in predictions]
    if "cot" in model_name:
        results_df["chain_of_thought_0"] = chains_of_thought
        results_df["conclusion_0"] = conclusions_col
        results_df["parsed_prediction_0"] = [pred[i] for pred in parsed_predictions]
    else:
        for i in range(n_candidates):
            results_df[f"raw_prediction_{i}"] = [raw_pred[i] for raw_pred in raw_predictions]
    results_df = pd.DataFrame(results_df)

    vllm_postfix = "-vllm" if use_vllm else ""

    if epoch != 0:
        if not os.path.exists(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}"):
            os.makedirs(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}")
        results_df.to_csv(f"predictions/{dataset}/{model_name}{rank_postfix}{quant_status}-epoch-{epoch}{vllm_postfix}/{target}_{modified_stance}.csv", index=False)
    else:
        if not os.path.exists(f"predictions/{dataset}/{model_name}{quant_status}{vllm_postfix}"):
            os.makedirs(f"predictions/{dataset}/{model_name}{quant_status}{vllm_postfix}")
        results_df.to_csv(f"predictions/{dataset}/{model_name}{quant_status}{vllm_postfix}/{target}_{modified_stance}.csv", index=False)


def inference_wiki(modified_stance, model_name="llama-3.2-1b-instruct", sentiment=False, use_vllm=True):

    max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    # load_in_4bit = False

    if load_in_4bit:
        quant_status = ""
    else:
        quant_status = "-unquantized"

    model_name_arg = get_model_name_arg(model_name)

    if use_vllm:
        sampling_params = SamplingParams(temperature=0, max_tokens=1024)
        llm = LLM(model=(model_name_arg + "-bnb-4bit").lower(), max_model_len=max_seq_length,
                  device="cuda")
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_arg, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    data = get_hf_wiki_data(modified_stance, use_web="web" in model_name, sentiment=sentiment)
    predictions = []
    raw_predictions = []
    answers_ls = []

    n_candidates = 1
    if use_vllm:
        outputs = llm.chat(data["conversations"], sampling_params)
        answers_ls = [[output.outputs[0].text] for output in outputs]
    else:
        for i in tqdm(range(len(data))):
            # if i > 0: # for debugging
                # break
            inputs = tokenizer.apply_chat_template(
                data[i]["conversations"], 
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")
            temperature = 0.001
            outputs = model.generate(input_ids = inputs, max_new_tokens = 1024,
                        use_cache = True, temperature = temperature,
                        num_return_sequences=n_candidates)
            
            if model_name.startswith("llama"):
                llama_outputs = tokenizer.batch_decode(outputs)
                answers = [parse_llama_outputs(output, cot=False) for output in llama_outputs]
            elif model_name.startswith("qwen"):
                qwen_outputs = tokenizer.batch_decode(outputs)
                answers = [parse_qwen_outputs(output, cot=False) for output in qwen_outputs]
            answers_ls.append(answers)

    datasets = ["cstance", "pstance2", "semeval"]
    targets_ls = [
        ["face_masks", "fauci", "school_closures", "stay_at_home_orders"],
        ["bernie", "trump", "biden"],
        ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"]
    ]

    results_df = {"dataset": [], "target": []}
    for dataset, targets in zip(datasets, targets_ls):
        for target in targets:
            results_df["dataset"].append(dataset)
            results_df["target"].append(target)

    for i, answers in enumerate(answers_ls):
        
        raw_predictions.append(answers)
        # eliminate non-alphabetical characters
        answers = ["".join(filter(str.isalpha, ans)) for ans in answers]
        sample_predictions = []
        if sentiment:
            for ans in answers:
                if ans.lower() in ["positive"]:
                    sample_predictions.append("POSITIVE")
                elif ans.lower() == "negative":
                    sample_predictions.append("NEGATIVE")
                elif ans.lower() in ["none", "neutral"] and results_df["dataset"][i] != "pstance2":
                    sample_predictions.append("NONE")
                else:
                    sample_predictions.append("INVALID")            
        else:
            for ans in answers:
                if ans.lower() in ["favor", "favour", "favorable", "favourable"]:
                    sample_predictions.append("FAVOR")
                elif ans.lower() == "against":
                    sample_predictions.append("AGAINST")
                elif ans.lower() in ["none", "neutral"] and results_df["dataset"][i] != "pstance2":
                    sample_predictions.append("NONE")
                else:
                    sample_predictions.append("INVALID")
        predictions.append(sample_predictions)

    for i in range(n_candidates):
        results_df[f"prediction_{i}"] = [pred[i] for pred in predictions]
    for i in range(n_candidates):
        results_df[f"raw_prediction_{i}"] = [raw_pred[i] for raw_pred in raw_predictions]
    results_df = pd.DataFrame(results_df) 

    vllm_postfix = "-vllm" if use_vllm else ""
    char_dir = "sentiment" if sentiment else "stance"
    file_name = "web" if "web" in model_name else "wiki"
    stance_postfix = "" if modified_stance == "default" else f"_{modified_stance}"

    if not os.path.exists(f"info-char/{char_dir}/{model_name}{quant_status}{vllm_postfix}"):
        os.makedirs(f"info-char/{char_dir}/{model_name}{quant_status}{vllm_postfix}")
    results_df.to_csv(f"info-char/{char_dir}/{model_name}{quant_status}{vllm_postfix}/{file_name}{stance_postfix}.csv", index=False)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local", help="Mode: local, finetune, inference, inference_wiki_stance, inference_wiki_sentiment")
    parser.add_argument("--dataset", default="cstance", help="Dataset: cstance, pstance2, semeval")
    parser.add_argument("--target", default="face_masks", help="Target: face_masks, fauci, school_closures, stay_at_home_orders, bernie, trump, biden, atheism, climate_change_is_a_real_concern, feminist_movement, hillary_clinton, legalization_of_abortion")
    parser.add_argument("--model", default="llama-3.2-1b-instruct", 
                        help="Model: llama and qwen models")
    parser.add_argument("--stance", default="default",
                        help="Stance: none, default, neutral, favour, against")
    parser.add_argument("--epoch", default=1, type=int,
                        help="Epoch: 1, 2, 3 for finetuning, 0 for inference without finetuning")
    parser.add_argument("--rank", default=32, type=int,
                        help="Rank: 16, 32 for finetuning")
    args = parser.parse_args()

    mode = args.mode
    dataset = args.dataset
    target = args.target
    model = args.model
    stance = args.stance
    epoch = args.epoch
    rank = args.rank

    datasets = ["cstance", "pstance2", "semeval"]
    targets_ls = [
        ["face_masks", "fauci", "school_closures", "stay_at_home_orders"],
        ["bernie", "trump", "biden"],
        ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"],
    ]

    # inference wiki
    if mode == "inference_wiki_stance":
        print(f"Running wiki stance inference for {stance} - {model}")
        inference_wiki(stance, model, sentiment=False, use_vllm=True)
    if mode == "inference_wiki_sentiment":
        print(f"Running wiki sentiment inference for {stance} - {model}")
        inference_wiki(stance, model, sentiment=True, use_vllm=True)

    # finetuning
    if mode == "finetune":
        print(f"Epoch {epoch}!")
        print(f"Running finetuning for {dataset} - {target} - {model} - rank {rank} - epoch {epoch}")
        finetune(dataset, target, model, rank=rank, epoch=epoch)

    # inference
    if mode == "inference":
        print(f"Epoch {epoch}!")
        print(f"Running inference for {dataset} - {model} - rank {rank} - epoch {epoch}")
        for target in targets_ls[datasets.index(dataset)]:
            inference(dataset, target, "none", model, rank=rank, epoch=epoch, use_vllm=False)
            time.sleep(4) # avoid GPU overload
            inference(dataset, target, "default", model, rank=rank, epoch=epoch, use_vllm=False)
            time.sleep(4) # avoid GPU overload

    # inference without finetuning
    if mode == "local":
        print(f"Running inference without finetuning for {dataset} - {target} - {stance} - {model}")
        inference(dataset, target, stance, model, epoch=0, use_vllm=True)