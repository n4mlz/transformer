import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model
import yaml

with open("models/config/model.yaml") as file:
   config = yaml.safe_load(file)

with open("models/config/tokipona.yaml") as file:
    lang = yaml.safe_load(file)

max_length = config["Tokenizer"]["max_length"]
tokenizer = model.Tokenizer(lang, max_length)

c = config["GPT"]
GPT = model.GPT(c["vocab_size"], c["emb_dim"], c["max_length"], c["num_blocks"], c["head"], c["hidden_dim"], c["dropout"])
GPT.load_state_dict(torch.load("weights/202305181107.pth", map_location=torch.device('cpu')))


def pred(prompt: str) -> str:
    prompt = tokenizer.encode(prompt)
    for _ in range(max_length - prompt.shape[0]):
        print(f"\r{tokenizer.decode(prompt)}", end="")
        out = prompt.view(1, prompt.shape[0], prompt.shape[1])
        out = GPT(out)
        out = torch.argmax(out, axis=-1)
        out = F.one_hot(out,num_classes=tokenizer.vocab_size)
        prompt = torch.cat((prompt, out), dim=0)
        if torch.argmax(out, axis=-1) == tokenizer.dict["[SEP]"]:
            break
    ans = tokenizer.decode(prompt)
    print("\r")
    return ans


if __name__ == '__main__':
    pred('toki!')