import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml  
from typing import Union


with open("models/config/model.yaml") as file:
    config = yaml.safe_load(file)

with open("models/config/tokipona.yaml") as file:
    lang = yaml.safe_load(file)


class Tokenizer:
    def __init__(self, lang: list[str], max_length: int) -> None:
        self.special= ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab = self.special + [",", ".", "!", "?", '"']
        self.vocab.extend(lang)
        self.vocab_size = len(self.vocab)
        self.max_length = max_length

        self.dict = {}
        for v, k in enumerate(self.vocab):
            self.dict[k] = v
    
    def encode(self, txt: Union[str, list[str]], padding: bool=False) -> torch.Tensor:
        def enc(txt: str) -> list[str]:
            s = "[CLS]" + txt.lower().replace(","," , ").replace("."," . [SEP] ").replace("!"," ! [SEP] ").replace("?"," ? [SEP] ").replace('"',' " ') + "  "
            s =  "".join([s[i] * (s[i] != " " or s[i+1] != " ") for i in range(len(s)-1)]).split(" ")
            if s[-1] != "[SEP]":
                s.append("[SEP]")
            s = [word if word in self.vocab else "[UNK]" for word in s]
            return s
        
        if type(txt) is str:
            s = enc(txt)
            length = self.max_length if padding else len(s)
            s += ["[PAD]"]*(length-len(s))
            s = [self.dict[word] for word in s]
        else:
            s = [enc(sent) for sent in txt]
            length = self.max_length if padding else min(max([len(sent) for sent in s]), self.max_length)
            s = [sent + ["[PAD]"]*(length-len(sent)) for sent in s]
            s = [[self.dict[word] for word in sent] for sent in s]
        
        s = torch.tensor(s)
        s = F.one_hot(s,num_classes=self.vocab_size)
        return s

    def decode(self, tns: torch.Tensor) -> Union[str, list[str]]:
        def dec(tns: list[int]) -> str:
            s = [self.vocab[word] for word in tns]
            s = [word for word in s if word not in self.special]
            s = " ".join(s).replace(" ,", ",").replace(" .", ".")
            return s
        
        s = torch.argmax(tns, dim=-1)
        if s.dim() == 1:
            s = dec(s.tolist())
        else:
            s = [dec(sent) for sent in s.tolist()]
        return s


class InputLayer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, max_length: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.emb = nn.Linear(vocab_size, self.emb_dim, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, max_length, emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, V) -> (B, N, D)
        z = self.emb(x.type(torch.float))
        z += self.pos_emb[:, :z.shape[-2]]
        return z


class Attention(nn.Module):
    def __init__(self, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.sqrt_dh = emb_dim ** 0.5
        self.attn_drop = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (, N, D) -> (, D, N)
        k_T = k.transpose(-1, -2)
        # (, N, D) x (, D, N) -> (, N, N)
        dots = (q @ k_T) / self.sqrt_dh

        attn = F.softmax(dots, dim=-1)
        # (, N, N) x (, N, D) -> (, N, D)
        attn = self.attn_drop(attn)

        out = attn @ v
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim:int, head:int, dropout:float) -> None:
        super().__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn = Attention(self.head_dim, dropout)

        self.attn_drop = nn.Dropout(dropout)
        self.w_o = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, _ = z.size()
        # (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # (B, h, N, D//h) -> (B, h, N, D//h)
        out = self.attn(q, k ,v)

        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim:int, head:int, hidden_dim:int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.msa = MultiHeadSelfAttention(emb_dim, head, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.ln1(z)) + z
        out = self.feedforward(self.ln2(out)) + out 
        return out
    

class OutputLayer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.ln = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.ln(z)
        out = F.softmax(out, dim=-1)
        return out


class GPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_length, num_blocks, head, hidden_dim, dropout) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.num_blocks = num_blocks
        self.head = head
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.inp = InputLayer(self.vocab_size, self.emb_dim, self.max_length)
        self.blocks = nn.Sequential(*[EncoderBlock(self.emb_dim, self.head, self.hidden_dim, self.dropout)for _ in range(self.num_blocks)])
        self.outp = OutputLayer(self.vocab_size, self.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inp(x)
        out = self.blocks(out)
        out = out[:, 0]
        pred = self.outp(out)
        return pred


if __name__ == '__main__':
    # print(f"counfig:{config}")
    # print(f"tokipona:{lang}")
    #model = MultiHeadSelfAttention(config["MultiHeadAttention"]["emb_dim"], config["MultiHeadAttention"]["head"], config["MultiHeadAttention"]["dropout"])
    # z = torch.randn(64, 32, 384)
    # print(model(z).shape)
    t = Tokenizer(lang, 64)
    a = t.encode('a, akesi ,anu')
    print(a)
    a = t.encode('a, akesi ,anu!')
    print(a)
    # b = t.encode(['a, akesi ,anu . ', 'a, akesi ,utala . e o'])
    # model = GPT(134, 12, 64, 8 ,4, 40, 0.1)
    # ans = model(b)
    # print(t.decode(ans))
    print(t.encode('a, a!', True).shape)
    print(t)