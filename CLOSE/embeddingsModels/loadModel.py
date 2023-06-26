from transformers import BioGptTokenizer, BioGptModel, BertModel, BertTokenizer

def loadBertModelAndTokenizer(model_dir: str, tokenizer_dir: str):
    """
    Load both the model and the tokanizer from saved file and return them

    :param model_dir: the directory to the model
    :param tokenizer_dir: the directory to the tokenizer
    """

    model = BertModel.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

    assert(isinstance(model, BertModel))
    assert(isinstance(tokenizer, BertTokenizer))

    return model, tokenizer

def loadBioGptModelAndTokenizer(model_dir: str, tokenizer_dir: str):
    """
    Load both the model and the tokanizer from saved file and return them

    :param model_dir: the directory to the model
    :param tokenizer_dir: the directory to the tokenizer
    """

    model = BioGptModel.from_pretrained(model_dir)
    tokenizer = BioGptTokenizer.from_pretrained(tokenizer_dir)

    assert(isinstance(model, BioGptModel))
    assert(isinstance(tokenizer, BioGptTokenizer))

    return model, tokenizer
