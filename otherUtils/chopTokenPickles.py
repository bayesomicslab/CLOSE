import pickle
import os


SPLIT_SIZE = 334338

tokenized_text = {}
with open(os.path.join(",", "tokenized_text.pkl"), "rb") as file:
    tokenized_text = pickle.load(file)
    file.close()

cnt = 0
block = 0
aux_tokenized = {}
for id in tokenized_text:
    if(cnt == SPLIT_SIZE):
        with open(os.path.join(".", f"tokenized_text_block_{block}.pkl"), "wb") as file:
            pickle.dump(aux_tokenized, file)
            file.close()

        aux_tokenized = {}
        cnt = 0
        block += 1

    aux_tokenized[id] = tokenized_text[id]
    cnt += 1

if cnt > 0:
    with open(os.path.join(".", f"tokenized_text_block_{block}.pkl"), "wb") as file:
        pickle.dump(aux_tokenized, file)
        file.close()
