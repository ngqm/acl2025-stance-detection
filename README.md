# Is External Information Useful for Stance Detection with LLMs?

This repository hosts the reproduction code for "Is External Information Useful for Stance Detection with LLMs?" (ACL Findings 2025) by [Quang Minh Nguyen](ngqm.github.io) (KAIST) and [Taegyoon Kim](taegyoon-kim.github.io) (KAIST).

[paper (coming soon)]

Updates:
- June 3, 2025: Fine-tuning and inference code for local LLMs (Qwen and Llama) are now available. We also provide Wikipedia and Web Search external information. More will be released soon.


## Installation

The main requirements are Unsloth and vLLM. You can install all packages we used during the development process by running the following commands:

```bash
conda create -n stance python=3.8
pip install -r requirements.txt
```

Additionally, in `scripts/config.py`, include a directory where you would like to store fine-tuned models.

Note that in earlier versions of `unsloth`, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` is used as-is when fine-tuning is conducted. However, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` is now mapped to `unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit` for fine-tuning. This leads to inconsistent performance. The same goes for all local models. In the case of this project, the latter results in more invalid predictions. Therefore to achieve the the performance level mentioned in our paper, you may manually access `unsloth/models/mapper.py` to change model mappings when conducting fine-tuning.

## Data

Please refer to the original papers for the datasets:
- [COVID-19-Stance (Glandt et. al., 2021)](https://aclanthology.org/2021.acl-long.127/)
- [P-Stance (Li et. al., 2021)](https://aclanthology.org/2021.findings-acl.208/)
- [SemEval-2016 Task 6 (Mohammad et. al., 2016)](https://aclanthology.org/S16-1003/)

After you have obtained the datasets, place them in `data/cstance`, `data/pstance`, and `data/semeval` directories, respectively. Please refer to `data/cstance/processing.ipynb`, `data/pstance/preprocessing.ipynb`, and `data/semeval/preprocessing.ipynb` for data preprocessing steps. Our preprocessing scripts are adapted from the code of [Li et. al. (2021)](https://aclanthology.org/2022.wassa-1.7/).

## Local LLMs

Our code supports fine-tuning and inference with various Qwen and Llama models. You can find more details in scripts/main.py

```
python scripts/main.py <mode>
```

where `<mode>` can be one of the following:
- `local`: Run inference without fine-tuning
- `finetune`: Fine-tune local LLMs on the stance detection task
- `inference`: Run inference using the fine-tuned LLMs
- `inference_wiki_stance`: Run stance inference on external information (Wikipedia or Web Search) alone
- `inference_wiki_sentiment`: Run sentiment inference on external information (Wikipedia or Web Search) alone
  
Before running this command, make sure to set the CUDA device, dataset, target, mode, epoch, and LoRA rank (where applicable) in `scripts/main.py`.


## Citation
If you use this code in your research, please cite our paper as follows:

```bibtex
@inproceedings{llm-stance-detection,
    title = "Is External Information Useful for Stance Detection with {LLM}s?",
    author = "Nguyen, Quang Minh and Kim, Taegyoon",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = july,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics"
}
```