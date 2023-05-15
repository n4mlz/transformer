"""
References: https://huggingface.co/spaces/Jayyydyyy/english-tokipona-translator
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("Jayyydyyy/m2m100_418m_tokipona")
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
LANG_CODES = {
    "English":"en",
    "toki pona":"tl"    
}

model.to(device)

def translate(text, src_lang, tgt_lang, candidates:int):
    """
    Translate the text from source lang to target lang
    """

    src = LANG_CODES.get(src_lang)
    tgt = LANG_CODES.get(tgt_lang)

    tokenizer.src_lang = src
    tokenizer.tgt_lang = tgt

    ins = tokenizer(text, return_tensors='pt').to(device)

    gen_args = {
            'return_dict_in_generate': True,
            'output_scores': True,
            'output_hidden_states': True,
            'length_penalty': 0.0,  # don't encourage longer or shorter output,
            'num_return_sequences': candidates,
            'num_beams':candidates,
            'forced_bos_token_id': tokenizer.lang_code_to_id[tgt]
        }
    

    outs = model.generate(**{**ins, **gen_args})
    output = tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)

    return '\n'.join(output)