import pickle
import os
import torch
from typing import Tuple


SPLIT_SIZE = 334338

WORK_DIR = "/mnt/research/aguiarlab/proj/CLOSE/nick_data"

tokenized_text: Tuple = ()
with open(os.path.join(WORK_DIR, "tokenized_text.pkl"), "rb") as file:
    tokenized_text = pickle.load(file)
    file.close()

cnt = 0
block = 0
aux_tokenized_ids = []
aux_tokenized_tokens = {"input_ids": [], "token_type_ids": [], "attention_mask": []}

for id, input_ids, token_type_ids, attention_mask in zip(tokenized_text[0], tokenized_text[1]["input_ids"], tokenized_text[1]["token_type_ids"], tokenized_text[1]["attention_mask"]):
    if(cnt == SPLIT_SIZE):
        aux_tokenized_tokens_tensored = {
            "input_ids": torch.stack(aux_tokenized_tokens["input_ids"]),
            "token_type_ids": torch.stack(aux_tokenized_tokens["token_type_ids"]),
            "attention_mask": torch.stack(aux_tokenized_tokens["attention_mask"])
        }
        with open(os.path.join(WORK_DIR, f"tokenized_text_block_{block}.pkl"), "wb") as file:
            #print((aux_tokenized_ids, aux_tokenized_tokens_tensored))
            pickle.dump((aux_tokenized_ids, aux_tokenized_tokens_tensored), file)
            file.close()

        
        aux_tokenized_ids = []
        aux_tokenized_tokens = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        cnt = 0
        block += 1

    aux_tokenized_ids.append(id)
    aux_tokenized_tokens["input_ids"].append(input_ids)
    aux_tokenized_tokens["token_type_ids"].append(token_type_ids)
    aux_tokenized_tokens["attention_mask"].append(attention_mask)
    cnt += 1

if cnt > 0:
    aux_tokenized_tokens_tensored = {
        "input_ids": torch.stack(aux_tokenized_tokens["input_ids"]),
        "token_type_ids": torch.stack(aux_tokenized_tokens["token_type_ids"]),
        "attention_mask": torch.stack(aux_tokenized_tokens["attention_mask"])
    }
    with open(os.path.join(WORK_DIR, f"tokenized_text_block_{block}.pkl"), "wb") as file:
        #print((aux_tokenized_ids, aux_tokenized_tokens_tensored))
        pickle.dump((aux_tokenized_ids, aux_tokenized_tokens_tensored), file)
        file.close()
