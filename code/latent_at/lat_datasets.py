import os
import json
from typing import List, Optional, Union
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sys
load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")


class LatentAdversarialTrainingDataset(Dataset):

    def __init__(self, hf_dataset):
        self.adv_tokens = hf_dataset["adv_tokens"]
        self.def_tokens = hf_dataset["def_tokens"]
        self.prompt_tokens = hf_dataset["prompt_tokens"]

        try:
            if "adv_strs" in hf_dataset.column_names:
                self.adv_strs = hf_dataset["adv_strs"]
                self.def_strs = hf_dataset["def_strs"]
                self.prompt_strs = hf_dataset["prompt_strs"]
        except:
            pass

        self.prompt_lengths = torch.tensor([len(x) for x in self.prompt_tokens])
        self.adv_lengths = torch.tensor([len(x) for x in self.adv_tokens])
        self.def_lengths = torch.tensor([len(x) for x in self.def_tokens])
        self.length = self.adv_lengths.shape[0]

        try:
            if "adv_labels" in hf_dataset.column_names:
                self.adv_labels = hf_dataset["adv_labels"]
                self.def_labels = hf_dataset["def_labels"]
                self.adv_indices = hf_dataset["adv_indices"]
                self.def_indices = hf_dataset["def_indices"]
        except:
            pass
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # if self.adv_strs exists, return it
        return_dict = {
            "adv_tokens": self.adv_tokens[idx],
            "def_tokens": self.def_tokens[idx],
            "prompt_tokens": self.prompt_tokens[idx],
            "adv_length": self.adv_lengths[idx],
            "def_length": self.def_lengths[idx],
            "prompt_length": self.prompt_lengths[idx],
        }
        if hasattr(self, 'adv_strs'):
            return_dict["adv_strs"] = self.adv_strs[idx]
            return_dict["def_strs"] = self.def_strs[idx]
            return_dict["prompt_strs"] = self.prompt_strs[idx]
        if hasattr(self, 'adv_labels'):
            return_dict["adv_labels"] = self.adv_labels[idx]
            return_dict["def_labels"] = self.def_labels[idx]
            return_dict["adv_indices"] = self.adv_indices[idx]
            return_dict["def_indices"] = self.def_indices[idx]
        return return_dict
                # "adv_strs": self.adv_strs[idx],
                # "def_strs": self.def_strs[idx],
                # "prompt_strs": self.prompt_strs[idx]


class LatentAdversarialTrainingDataCollator:
    def __init__(self, pad_token_id, truncate_length=None):
        assert pad_token_id is not None, "pad_token_id must be specified"

        self.pad_token_id = pad_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch):
        B = len(batch)

        prompt_lengths = []
        adv_prompt_lengths = []
        def_prompt_lengths = []
        seprate_prompt_length = []
        seprate_adv_length = []
        seprate_def_length = []

        for i in range(B):
            prompt_lengths.append(batch[i]["prompt_length"])
            adv_prompt_lengths.append(batch[i]["prompt_length"] + batch[i]["adv_length"])
            def_prompt_lengths.append(batch[i]["prompt_length"] + batch[i]["def_length"])
        
        pad_length = max(adv_prompt_lengths + def_prompt_lengths)

        adv_prompt_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        def_prompt_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        adv_padd_prompt_tokens = torch.zeros(B, pad_length + 20, dtype=torch.long)
        def_padd_prompt_tokens = torch.zeros(B, pad_length + 20, dtype=torch.long)
        padd_prompt_mask = torch.zeros(B, pad_length + 20, dtype=torch.bool)
        prompt_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        adv_padd_labels_mask = torch.zeros(B, pad_length + 20, dtype=torch.bool)
        def_padd_labels_mask = torch.zeros(B, pad_length + 20, dtype=torch.bool)
        adv_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        def_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)

        for i in range(B):
            adv_padd_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"][:-6] +[1738] * 20 + batch[i]["prompt_tokens"][-6:] + batch[i]["adv_tokens"] + [self.pad_token_id] * (pad_length - adv_prompt_lengths[i] ))
            def_padd_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"][:-6] +[1738] * 20 + batch[i]["prompt_tokens"][-6:] + batch[i]["def_tokens"] + [self.pad_token_id] * (pad_length - def_prompt_lengths[i] ))
            adv_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"] + batch[i]["adv_tokens"] + [self.pad_token_id] * (pad_length - adv_prompt_lengths[i]))
            def_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"] + batch[i]["def_tokens"] + [self.pad_token_id] * (pad_length - def_prompt_lengths[i]))
            seprate_prompt_length.append( (4,len(batch[i]["prompt_tokens"])-6 + 20,len(batch[i]["prompt_tokens"]) + 20,pad_length.item() ))
            padd_prompt_mask[i, len(batch[i]["prompt_tokens"])-6 :len(batch[i]["prompt_tokens"])-6 + 20] = True
            prompt_mask[i, :prompt_lengths[i]] = True
            adv_labels_mask[i, prompt_lengths[i]:adv_prompt_lengths[i]] = True
            def_labels_mask[i, prompt_lengths[i]:def_prompt_lengths[i]] = True
            adv_padd_labels_mask[i, prompt_lengths[i]+20: adv_prompt_lengths[i]+20] = True
            def_padd_labels_mask[i, prompt_lengths[i]+ 20:def_prompt_lengths[i] + 20] = True

        if self.truncate_length is not None:
            if any([prompt_length > self.truncate_length for prompt_length in prompt_lengths]):
                print(f"WARNING: Prompt length (at least one of {prompt_lengths}) is less than truncate length ({self.truncate_length})")

            adv_prompt_tokens = adv_prompt_tokens[:, :self.truncate_length]
            def_prompt_tokens = def_prompt_tokens[:, :self.truncate_length]
            prompt_mask = prompt_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]

        
        if "adv_strs" in batch[0]:
            return {
                "adv_tokens": adv_prompt_tokens,
                "def_tokens": def_prompt_tokens,
                "prompt_mask": prompt_mask,
                "adv_labels_mask": adv_labels_mask,
                "def_labels_mask": def_labels_mask,
                "adv_strs": [x["adv_strs"] for x in batch],
                "def_strs": [x["def_strs"] for x in batch],
                "prompt_strs": [x["prompt_strs"] for x in batch]
            }
        return {
            "adv_tokens": adv_prompt_tokens,
            "def_tokens": def_prompt_tokens,
            "adv_padd_prompt_tokens": adv_padd_prompt_tokens,
            "def_padd_prompt_tokens": def_padd_prompt_tokens,
            "adv_padd_labels_mask": adv_padd_labels_mask,
            "def_padd_labels_mask": def_padd_labels_mask,
            "prompt_mask": prompt_mask,
            "adv_labels_mask": adv_labels_mask,
            "def_labels_mask": def_labels_mask,
            "prompt_length": seprate_prompt_length,
            "padd_prompt_mask": padd_prompt_mask
            # "adv_strs": [x["adv_strs"] for x in batch],
            # "def_strs": [x["def_strs"] for x in batch],
            # "prompt_strs": [x["prompt_strs"] for x in batch]
        }


def apply_chat_formatting(
    tokenizer,
    prompt,
    def_completion,
    adv_completion,
    use_tokenizer_template,
    system_prompt,
    custom_prompt_template,
    custom_completion_template
):
    if use_tokenizer_template:
        if system_prompt is not None:
            prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            prompt_messages = [{"role": "user", "content": prompt}]
        
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    else:
        if system_prompt is not None:
            prompt_str = custom_prompt_template.format(system_prompt=system_prompt, prompt=prompt)
        else:
            prompt_str = prompt
    
    if custom_completion_template is not None:
        adv_str = custom_completion_template.format(completion=adv_completion)
        def_str = custom_completion_template.format(completion=def_completion)
    else:
        adv_str = adv_completion
        def_str = def_completion
    
    return prompt_str, adv_str, def_str


def process_generic_chat_dataset(
    tokenizer,
    dataset="Baidicoot/comic_villain_completions",
    prompt_column="prompt",
    adv_column="adv_completion",
    def_column="clean_completion",
    use_tokenizer_template=True,
    custom_prompt_template=None,
    custom_completion_template=None,
    system_prompt=None,
    system_prompt_column=None,
    filter_len=None,
    num_adv_words=None,
    map_fn=None,
    add_eos_token=False,
    **dataset_kwargs,
):
    # loader for generic datasets of the form (prompt, positive_completion, negative_completion)
    assert not (system_prompt is not None and system_prompt_column is not None), "Only one of system_prompt and system_prompt_column can be specified"

    #dataset = load_dataset(dataset, **dataset_kwargs)
    if dataset == "benign-dataset":
        print('load benign')
        dataset = load_dataset('parquet', data_files=f"/root/autodl-tmp/base/benign.parquet",split='train')
        #print(dataset['train'].to_pandas().head())
        #print(set(dataset.column_names) - {"prompt", "adv_completion", "def_completion"})
    elif dataset == 'harmful-dataset':
        print('load harmful')
        dataset = load_dataset('parquet', data_files="./at_final/datasets/harmful.parquet",split='train')
        #print(dataset['train'].to_pandas().head())
       # print(set(dataset.column_names) - {"prompt", "adv_completion", "def_completion"})
    print(f"Initial dataset columns: {dataset.column_names}")
    # Initial dataset columns: {'train': ['prompt', 'rejected', 'chosen']}
    if prompt_column != "prompt":
        dataset = dataset.rename_column(prompt_column, "prompt")

    if adv_column != "adv_completion":
        if adv_column is None:
             dataset = dataset.map(lambda x: {"adv_completion": "not available"})
        else:
            dataset = dataset.rename_column(adv_column, "adv_completion")
    
    if def_column != "def_completion":
        dataset = dataset.rename_column(def_column, "def_completion")

    if system_prompt_column is not None:
        dataset = dataset.rename_column(system_prompt_column, "system_prompt")
    
    if map_fn is not None:
        dataset = dataset.map(map_fn, batched=True)
    
    def preprocess_example_batch(examples):
        for i in range(len(examples["prompt"])):
            if system_prompt_column is not None:
                _system_prompt = examples["system_prompt"][i]
            elif system_prompt is not None:
                _system_prompt = system_prompt
            else:
                _system_prompt = None
                
            prompt, adv_completion, def_completion= apply_chat_formatting(
                tokenizer=tokenizer,
                prompt=examples["prompt"][i],
                def_completion=examples["def_completion"][i],
                adv_completion=examples["adv_completion"][i],
                use_tokenizer_template=use_tokenizer_template,
                system_prompt=_system_prompt,
                custom_prompt_template=custom_prompt_template,
                custom_completion_template=custom_completion_template
            )
            
            examples["prompt"][i] = prompt
            
            if num_adv_words is None:
                examples["adv_completion"][i] = adv_completion
            else:
                examples["adv_completion"][i] = " ".join(adv_completion.split(" ")[:num_adv_words])

            if add_eos_token:
                examples["def_completion"][i] = def_completion + tokenizer.eos_token
            else:
                examples["def_completion"][i] = def_completion
        
        return examples

    dataset = dataset.map(
        preprocess_example_batch,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in {"prompt", "adv_completion", "def_completion"}]
    )

    def remove_duplicate_bos_batched(batch_of_sequences, bos_token):
        def process_sequence(sequence):
            # Find the index of the last BOS token at the start of the sequence
            last_bos_index = 0
            for i, token in enumerate(sequence):
                if token != bos_token:
                    break
                last_bos_index = i

            # Return a single BOS token followed by the rest of the sequence
            return [bos_token] + sequence[last_bos_index + 1:]

        return [process_sequence(seq) for seq in batch_of_sequences]

    def tokenize_batch(examples):
        examples["prompt_tokens"] = remove_duplicate_bos_batched(
            tokenizer(examples["prompt"], add_special_tokens=True).input_ids,
            tokenizer.bos_token_id
        )
        examples["adv_tokens"] = tokenizer(examples["adv_completion"], add_special_tokens=False).input_ids
        examples["def_tokens"] = tokenizer(examples["def_completion"], add_special_tokens=False).input_ids
        
        return examples
    
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns={"prompt", "adv_completion", "def_completion"}
    )

    if filter_len is not None:
        start_len = len(dataset)
        dataset = dataset.filter(
            lambda x: len(x["prompt_tokens"]) + max(len(x["adv_tokens"]), len(x["def_tokens"])) <= filter_len
        )
        end_len = len(dataset)
        print(f"Filtered out {(start_len - end_len) / start_len * 100:.2f}% of the dataset")

    return LatentAdversarialTrainingDataset(dataset)


