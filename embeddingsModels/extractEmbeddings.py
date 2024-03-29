from typing import List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

# This tokenizes text for bert models to use. Runs on CPU. Look into BertTokenizerFast for faster gpu alternatives with downside of consuming more memory
def tokenizeText(tokenizer: PreTrainedTokenizer, data: Tuple[List, List]):
    """
    Tokenize the data passed in

    :param tokenizer: the tokenizer to be used for the data before passing into the model
    """

    ids = data[0]
    data_str = data[1]

    print("STARTING TOKENIZING")
    tokenized_text = tokenizer(data_str, truncation=True, padding="max_length",  return_tensors="pt")
    print("FINISHED TOKENIZING")

    return ids, tokenized_text


def extractEmbeddings(model: PreTrainedModel, data: Tuple[List, Dict]):
    """
    Extract the embeddings for the data from the bert model

    :param model: the model that will be used for embedding creation
    """

    ids = data[0]
    data_tokenized = data[1]

    print("MODEL RUNNING")
    embeddings = model(**data_tokenized).last_hidden_state
    print("FINISHED MODEL RUNNING")

    return ids, embeddings
