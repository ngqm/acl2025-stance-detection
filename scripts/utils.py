import numpy as np
import pandas as pd
import os
import json
import pickle
from datasets import Dataset
from sklearn.metrics import f1_score
from scipy import stats
import re


def get_shorthand(target):

    translation = {
        "Face Masks": "face_masks", "Anthony Fauci": "fauci", "School Closures": "school_closures", "Stay at Home Orders": "stay_at_home_orders",
        "Bernie Sanders": "bernie", "Joe Biden": "biden", "Donald Trump": "trump",
        "Atheism": "atheism", "Climate Change is a Real Concern": "climate_change_is_a_real_concern", "Feminist Movement": "feminist_movement", "Hillary Clinton": "hillary_clinton", "Legalization of Abortion": "legalization_of_abortion",
    }
    return translation.get(target, target)


def get_messages(text, target, wiki="", remove_none=False, cot=False, fewshot=False, stance="default", use_web=False):

    if remove_none:
        options_message = "Options: [FAVOR, AGAINST]"
    else:
        options_message = "Options: [FAVOR, AGAINST, NONE]"

    shorthand = get_shorthand(target)
    stance_postfix = "" if stance == "default" else f"_{stance}"

    messages = []
    if fewshot:
        if target in ["Face Masks", "Anthony Fauci", "School Closures", "Stay at Home Orders"]:
            if wiki == "":
                fewshot_path = f"few-shot/cstance/{shorthand}.json"
            elif use_web:
                fewshot_path = f"few-shot/cstance/{shorthand}_web.json"
            else:
                fewshot_path = f"few-shot/cstance/{shorthand}_wiki{stance_postfix}.json"
        elif target in ["Bernie Sanders", "Joe Biden", "Donald Trump"]:
            if wiki == "":
                fewshot_path = f"few-shot/pstance2/{shorthand}.json"
            elif use_web:
                fewshot_path = f"few-shot/pstance2/{shorthand}_web.json"
            else:
                fewshot_path = f"few-shot/pstance2/{shorthand}_wiki{stance_postfix}.json"
        elif target in ["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion"]:
            if wiki == "":
                fewshot_path = f"few-shot/semeval/{shorthand}.json"
            elif use_web:
                fewshot_path = f"few-shot/semeval/{shorthand}_web.json"
            else:
                fewshot_path = f"few-shot/semeval/{shorthand}_wiki{stance_postfix}.json"
        # retrieve in-context examples
        with open(fewshot_path, 'r') as f:
            fewshot_examples = json.load(f)
        messages += fewshot_examples

    if wiki != "":
        if cot:
            messages += [
                {"role": "user", "content": f"You are given the following text: {text}"},
                {"role": "user", "content": f"What is the stance of the text towards the target '{target}'?"},
                {"role": "user", "content": f"Integrate the following external information and do NOT automatically adopt the stance of it: {wiki}"},
                {"role": "user", "content": options_message},
                {"role": "assistant", "content": "Let's think step by step."}
            ]
        else:
            messages += [
                {"role": "user", "content": f"You are given the following text: {text}"},
                {"role": "user", "content": f"What is the stance of the text towards the target '{target}'?"},
                {"role": "user", "content": f"The following information can be helpful: {wiki}"},
                {"role": "user", "content": options_message},
                {"role": "user", "content": "Do not explain. Just provide the stance in a single word."}
            ]
    else:
        if cot:
            messages += [
                {"role": "user", "content": f"You are given the following text: {text}"},
                {"role": "user", "content": f"What is the stance of the text towards the target '{target}'?"},
                {"role": "user", "content": options_message},
                {"role": "assistant", "content": "Let's think step by step."}
            ]
        else:
            messages += [
                {"role": "user", "content": f"You are given the following text: {text}"},
                {"role": "user", "content": f"What is the stance of the text towards the target '{target}'?"},
                {"role": "user", "content": options_message},
                {"role": "user", "content": "Do not explain. Just provide the stance in a single word."}
            ]
    return messages


def get_wiki_messages(target, wiki, remove_none=False, cot=False, sentiment=False):

    if sentiment: # sentiment
        if remove_none:
            options_message = "Options: [POSITIVE, NEGATIVE]"
        else:
            options_message = "Options: [POSITIVE, NEGATIVE, NONE]"
    else: # stance
        if remove_none:
            options_message = "Options: [FAVOR, AGAINST]"
        else:
            options_message = "Options: [FAVOR, AGAINST, NONE]"

    if cot:
        if sentiment:
            messages=[
                {"role": "user", "content": f"You are given the following text: {wiki}"},
                {"role": "user", "content": f"What is the sentiment of the text?"},
                {"role": "user", "content": options_message},
                {"role": "assistant", "content": "Let's think step by step."}
            ]
        else:
            messages=[
                {"role": "user", "content": f"You are given the following text: {wiki}"},
                {"role": "user", "content": f"What is the stance of the text towards the target '{target}'?"},
                {"role": "user", "content": options_message},
                {"role": "assistant", "content": "Let's think step by step."}
            ]
    else:
        if sentiment:
            messages=[
                {"role": "user", "content": f"You are given the following text: {wiki}"},
                {"role": "user", "content": f"What is the sentiment of the text?"},
                {"role": "user", "content": options_message},
                {"role": "user", "content": "Do not explain. Just provide the sentiment in a single word."}
            ]
        else:
            messages=[
                {"role": "user", "content": f"You are given the following text: {wiki}"},
                {"role": "user", "content": f"What is the stance of the text towards the target '{target}'?"},
                {"role": "user", "content": options_message},
                {"role": "user", "content": "Do not explain. Just provide the stance in a single word."}
            ]
    return messages


def get_cpu_data(dataset, split="test"):
    """
    Get test data for a given dataset.

    :param dataset: str, dataset. Either "covid19-stance" or "pstance"
    """
    if dataset == "cstance":
        targets = ["face_masks", "fauci", "school_closures", "stay_at_home_orders"]
    elif dataset == "pstance2":
        targets = ["bernie", "biden", "trump"]
    elif dataset == "semeval":
        targets = ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"]
    
    test_data = {}
    for target in targets:
        if dataset == "cstance":
            test_data[target] = pd.read_csv(f"data/cstance/{target}_{split}.csv")
        elif dataset == "pstance2":
            test_data[target] = pd.read_csv(f"data/pstance/raw_{split}_{target}.csv")
            test_data[target] = test_data[target][test_data[target]["Stance"] != "NONE"]
            test_data[target].reset_index(drop=True, inplace=True)
        elif dataset == "semeval":
            test_data[target] = pd.read_csv(f"data/semeval/{target}_{split}.csv")
    return test_data


def get_hf_data(dataset, split, target, modified_stance, cot, use_web=False, chains_of_thought=None, fewshot=False):

    if dataset == "cstance":
        data_df = pd.read_csv(f"data/cstance/{target}_{split}.csv")
    elif dataset == "pstance2":
        data_df = pd.read_csv(f"data/pstance/raw_{split}_{target}.csv")
        data_df = data_df[data_df["Stance"] != "NONE"]
        data_df.reset_index(drop=True, inplace=True)
    elif dataset == "semeval":
        data_df = pd.read_csv(f"data/semeval/{target}_{split}.csv")
    
    if target in ["bernie", "biden", "trump"]:
        wiki_name = {"bernie": "bernie sanders", "biden": "joe biden", "trump": "donald trump"}[target]
    else:
        wiki_name = target
    wiki = get_biased_wiki(dataset, modified_stance, use_web=use_web)[wiki_name]
    # get messages based on data_df.Tweet, data_df.Target, and wiki
    messages_ls = []
    for i in range(len(data_df)):
        # if i > 0:  # for debugging
            # break
        text = data_df.Tweet[i]
        remove_none = (dataset == "pstance2")
        messages = get_messages(text, data_df.Target[i], wiki, remove_none=remove_none, cot=cot, fewshot=fewshot, use_web=use_web, stance=modified_stance)
        if chains_of_thought is not None:
            messages += [{"role": "assistant", "content": chains_of_thought[i]}]
            if remove_none:
                messages += [{"role": "assistant", "content": "Therefore, between FAVOR and AGAINST, the final answer is "}]
            else:
                messages += [{"role": "assistant", "content": "Therefore, among FAVOR, AGAINST, and NONE, the final answer is "}]
        if split == "train":
            messages += [{"role": "assistant", "content": data_df.Stance[i]}]
            messages_ls.append({"conversations": messages})
        else:
            messages_ls.append({"conversations": messages, "ground_truth": data_df.Stance[i]})

    data = Dataset.from_list(messages_ls)
    
    return data


def get_hf_wiki_data(modified_stance, cot=False, use_web=False, sentiment=False):

    datasets = ["cstance", "pstance2", "semeval"]
    targets_ls = [
        ["face_masks", "fauci", "school_closures", "stay_at_home_orders"],
        ["bernie", "trump", "biden"],
        ["atheism", "climate_change_is_a_real_concern", "feminist_movement", "hillary_clinton", "legalization_of_abortion"]
    ]

    messages_ls = []
    for dataset, targets in zip(datasets, targets_ls):
        remove_none = (dataset == "pstance2")
        for target in targets:
            if dataset == "cstance":
                data_df = pd.read_csv(f"data/cstance/{target}_test.csv")
            elif dataset == "pstance2":
                data_df = pd.read_csv(f"data/pstance/raw_test_{target}.csv")
                data_df = data_df[data_df["Stance"] != "NONE"]
                data_df.reset_index(drop=True, inplace=True)
            elif dataset == "semeval":
                data_df = pd.read_csv(f"data/semeval/{target}_test.csv")
        
            if dataset == "pstance2":
                wiki_name = {"bernie": "bernie sanders", "biden": "joe biden", "trump": "donald trump"}[target]
            else:
                wiki_name = target
            wiki = get_biased_wiki(dataset, modified_stance, use_web=use_web)[wiki_name]
            # get messages based on data_df.Tweet, data_df.Target, and wiki
            Target = data_df.Target[0]
            messages = get_wiki_messages(Target, wiki, remove_none=remove_none, cot=cot, sentiment=sentiment)
            messages_ls.append({"conversations": messages})
    
    data = Dataset.from_list(messages_ls)
    
    return data


def parse_llama_outputs(input_text, cot=False):
    # Regex to match segments based on <|start_header_id|role<|end_header_id|> and their content
    pattern = r"<\|start_header_id\|>(.*?)<\|end_header_id\|>\n\n(.*?)(?=<\|eot_id\|>|$)"
    matches = re.findall(pattern, input_text, re.DOTALL)
    # Convert matches to Hugging Face format
    return matches[-1][1]
        

def parse_qwen_outputs(input_text, cot=False):
    matches = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', input_text, re.DOTALL)
    return matches[-1]


# get biased wiki
def get_biased_wiki(dataset, stance, use_web=False):
    """Get biased wiki for a given dataset and stance.

    :param dataset: str, dataset. Either "covid", "pstance", or "semeval"
    :param stance: str, stance. Either "none", "default", "favour", "against", or "neutral"
    """
    info_source = "web" if use_web else "wiki"
    if stance == "default":

        if dataset == "cstance":
            with open(f'data/cstance/{info_source}_dict.pkl', 'rb') as f:
                biased_wiki_dict = pickle.load(f)
        elif dataset == "pstance2":
            with open(f'data/pstance/{info_source}_dict.pkl', 'rb') as f:
                biased_wiki_dict = pickle.load(f)
        elif dataset == "semeval":
            with open(f'data/semeval/{info_source}_dict.pkl', 'rb') as f:
                biased_wiki_dict = pickle.load(f)
        return biased_wiki_dict

    if dataset == "cstance":
        path = f'data/cstance/{info_source}_dict_{stance}.pkl'
    elif dataset == "pstance2":
        path = f'data/pstance/{info_source}_dict_{stance}.pkl'
    elif dataset == "semeval":
        path = f'data/semeval/{info_source}_dict_{stance}.pkl'
    with open(path, 'rb') as f:
        biased_wiki_dict = pickle.load(f)
    return biased_wiki_dict


if __name__=="__main__":

    pass