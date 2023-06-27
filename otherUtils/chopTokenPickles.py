import pickle
import os
from typing import Tuple


SPLIT_SIZE = 334338

tokenized_text: Tuple = ()
with open(os.path.join(".", "tokenized_text.pkl"), "rb") as file:
    tokenized_text = pickle.load(file)
    file.close()

print(tokenized_text)

cnt = 0
block = 0
aux_tokenized_ids = []
aux_tokenized_tokens = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
for id, input_ids, token_type_ids, attention_mask in zip(tokenized_text[0], tokenized_text[1]["input_ids"], tokenized_text[1]["token_type_ids"], tokenized_text[1]["attention_mask"]):
    if(cnt == SPLIT_SIZE):
        with open(os.path.join(".", f"tokenized_text_block_{block}.pkl"), "wb") as file:
            print((aux_tokenized_ids, aux_tokenized_tokens))
            pickle.dump((aux_tokenized_ids, aux_tokenized_tokens), file)
            file.close()

        aux_tokenized = ()
        cnt = 0
        block += 1

    aux_tokenized_ids.append(id)
    aux_tokenized_tokens["input_ids"].append(input_ids)
    aux_tokenized_tokens["token_type_ids"].append(token_type_ids)
    aux_tokenized_tokens["attention_mask"].append(attention_mask)
    cnt += 1

if cnt > 0:
    with open(os.path.join(".", f"tokenized_text_block_{block}.pkl"), "wb") as file:
        print((aux_tokenized_ids, aux_tokenized_tokens))
        pickle.dump((aux_tokenized_ids, aux_tokenized_tokens), file)
        file.close()
