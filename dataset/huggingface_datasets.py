from torch.utils.data import DataLoader
import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer
import os

ROOT = ""  # Path to the root directory of the dataset

task_to_keys = {
    "snli": ("premise", "hypothesis"),
    "mnli": ("text1", "text2"),
    "sick": ("sentence_A", "sentence_B"),

    "qnli": ("text1", "text2"),
    "rte": ("text1", "text2"),
    "scitail": ("premise", "hypothesis"),
}

task_ids = {
    "snli": "stanfordnlp/snli",
    "mnli": "SetFit/mnli",
    "sick": "sick",

    "qnli": "SetFit/qnli",
    "rte": "SetFit/rte",
    "scitail": "allenai/scitail",
}


class HuggingFaceDataset:
    def __init__(self,
                 location=None,
                 task=None,
                 model_name_or_path=None,
                 batch_size=32,
                 num_workers=16):
        self.batch_size = batch_size
        self.num_workers = num_workers
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        cache_dir = location if location else "./data"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_name = f"{task}_tokenized_{model_name_or_path.replace('/', '_')}"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            self.tokenized_datasets = load_from_disk(cache_path)
        else:
            dataset = datasets.load_dataset(task_ids[task], cache_dir=ROOT, trust_remote_code=True)
            sentence1_key, sentence2_key = task_to_keys[task]

            def tokenize_function(examples):
                args = (
                    (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
                )
                outputs = self.tokenizer(*args, truncation=True, max_length=2000)
                return outputs

            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=[t for t in dataset['train'].column_names if t != "label"]
            )

            self.tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            self.tokenized_datasets = self.tokenized_datasets.filter(lambda example: example["labels"] == 0 or example["labels"] == 1 or example["labels"] == 2)
            
            print(f"Saving dataset to {cache_path}")
            self.tokenized_datasets.save_to_disk(cache_path)

        if "train" in self.tokenized_datasets:
            self.train_loader = DataLoader(self.tokenized_datasets["train"], shuffle=True, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)
        if "test" in self.tokenized_datasets:
            self.test_loader = DataLoader(self.tokenized_datasets["test"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)
        if "validation" in self.tokenized_datasets:
            self.val_loader = DataLoader(self.tokenized_datasets["validation"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers=num_workers)

    def collate_fn(self, examples):
        return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")
