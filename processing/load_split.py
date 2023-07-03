from typing import Callable, Dict, List, Tuple

from transformers import PreTrainedModel
import torch
import gc


def extractEmbeddingsLoadSplit(data: Tuple[List, Dict], model: PreTrainedModel, run: Callable, split_size: int = 1000):
    """
    Split the load on tokenized data to be run on a function

    :param data: the data to be run with load splitting
    :param model: the model to be passed into runnable
    :param run: the function to be called with model and dat for the extraction of embeddings 
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    id_embeddings = []
    extracted_embeddings = []

    id_batch = []
    tokenized_batch = {}
    cnt = 0
    cur_batch = 0

    for id, input_ids, token_type_ids, attention_mask in zip(data[0], data[1]["input_ids"], data[1]["token_type_ids"], data[1]["attention_mask"]):
        if cnt == split_size:
            print(f"MOVING BATCH {cur_batch} TO GPU")
            tokenized_batch_to_device = {
                "input_ids": torch.stack(tokenized_batch["input_ids"]).to(device),
                "token_type_ids": torch.stack(tokenized_batch["token_type_ids"]).to(device),
                "attention_mask": torch.stack(tokenized_batch["attention_mask"]).to(device)
            }
            print(f"FINISHED MOVING BATCH {cur_batch} TO GPU")

            print(f"RUNNING BATCH {cur_batch} EXTRACT EMBEDDINGS")

            embeddings_batch = run(model, (id_batch, tokenized_batch_to_device))
            print(f"FINISHED RUNNING BATCH {cur_batch} EXTRACT EMBEDDINGS")

            print(f"MOVING BATCH {cur_batch} TO CPU")
            for i, emb in embeddings_batch:
                id_embeddings.append(i)
                extracted_embeddings.append(emb.to("cpu"))
            print(f"FINISHED MOVING BATCH {cur_batch} TO CPU")

            id_batch = []
            tokenized_batch = {}

            cnt = 0
            cur_batch = 1

            print(f"COLLECTING GARBAGE")
            gc.collect()
            print(f"FINISHED COLLECTING GARBAGE")

        id_batch.append(id)
        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["token_type_ids"].append(token_type_ids)
        tokenized_batch["attention_mask"].append(attention_mask)
        cnt += 1


        
    return id_embeddings, torch.stack(extracted_embeddings)
