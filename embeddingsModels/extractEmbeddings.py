from typing import List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch


def extractEmbeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, data: List[Dict[str, str]]):
    """
    Extract the embeddings for the data from the bert model

    :param model: the model that will be used for embedding creation
    :param tokenizer: the tokenizer to be used for the data before passing into the model
    """

    ids = [i["id"] for i in data]
    data_str = [i["text"] for i in data]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print("STARTING TOKENIZING")
    tokenized_text = tokenizer(data_str, truncation=True, padding="max_length",  return_tensors="pt")
    print("FINISHED TOKENIZING")
    tokenized_text = tokenized_text.to(device)
    print("TOKENIZING TO CUDA")
    embeddings = model(**tokenized_text).last_hidden_state
    print("MODEL RUNNING")

    tokens_and_embeddings = []
    for sample_emb, sample_tok, sample_id in zip(embeddings, tokenized_text["input_ids"], ids): #type: ignore
        sample_embedding = {
            "id": sample_id,
            "tokens": tokenizer.convert_ids_to_tokens(sample_tok),
            "token_ids": sample_tok,
            "embeddings": sample_emb
        }

        tokens_and_embeddings.append(sample_embedding)

    return tokens_and_embeddings, embeddings
