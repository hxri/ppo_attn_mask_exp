from datasets import load_dataset
from torch.utils.data import Dataset
from .rewards import format_gsm8k_prompt


class GSM8KDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None, max_prompt_len: int = 256):
        raw = load_dataset("gsm8k", "main", split=split)
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.examples = [
            {
                "question": ex["question"],
                "answer": ex["answer"],
                "prompt": format_gsm8k_prompt(ex["question"]),
            }
            for ex in raw
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.tokenizer is None:
            return ex
        tokens = self.tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=self.max_prompt_len,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "answer": ex["answer"],
            "prompt": ex["prompt"],
        }


def load_gsm8k(split: str = "train", tokenizer=None, max_prompt_len: int = 256) -> GSM8KDataset:
    return GSM8KDataset(split=split, tokenizer=tokenizer, max_prompt_len=max_prompt_len)


def collate_fn(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answers": [b["answer"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
    }
