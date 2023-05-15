"""
Refernces: https://huggingface.co/datasets/OpenAssistant/oasst1
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
import tokipona_dataset.translator
from models import model
import yaml

with open("models/config/model.yaml") as file:
    config = yaml.safe_load(file)

with open("models/config/tokipona.yaml") as file:
    lang = yaml.safe_load(file)


class MakeDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)
 
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.xs[idx])
        y = torch.FloatTensor(self.ys[idx])
        return x, y


def make_datasets(dataset):
    max_length = config["Tokenizer"]["max_length"]
    tokenizer = model.Tokenizer(lang, max_length)
    x_dataset = []
    y_dataset = []

    def sent_to_dataset(sent: torch.Tensor, idx: int) -> tuple[torch.Tensor]:
        x = torch.argmax(sent, dim=-1)
        if x[idx] == tokenizer.dict["[PAD]"]:
            return None, None
        y = x[idx].clone()
        for i in range(x.shape[-1] - idx):
            x[i + idx] = tokenizer.dict["[PAD]"]
        x = F.one_hot(x,num_classes=tokenizer.vocab_size)
        y = F.one_hot(y,num_classes=tokenizer.vocab_size)
        return x, y
    
    for data in dataset:
        if data["lang"] == "en":
            sents = translator.translate(data["text"], "English", "toki pona", 10).split("\n")
            for sent in sents:
                sent = tokenizer.encode(sent, True)
                for  idx in range(max_length - 1):
                    x, y  = sent_to_dataset(sent, idx)
                    if x != None:
                        x_dataset.append(x)
                        y_dataset.append(y)
    
    tokipona_dataset = MakeDataset(x_dataset, y_dataset)
    return tokipona_dataset


if __name__ == "__main__":
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds['train']      # len(train)=84437 (95%)
    val = ds['validation']   # len(val)=4401 (5%)
    train_dataset = make_datasets(train)
    val_dataset = make_datasets(val)
