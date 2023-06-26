import pandas as pd

def loadDataset(file: str, text_column: str, id_column: str):
    """
    Read the dataset and preprocess

    :param file: the file that contains the dataset
    :param text_column: the column with the text
    :param id_column: the column with the ids
    """

    df = pd.read_csv(file)
    
    df["text"] = df[text_column]
    df["id"] = df[id_column]

    return df

def loadCloseDataset(file: str):
    """
    Read the dataset and preprocess

    :param file: the file that contains the complaint data
    """

    df = loadDataset(file, "abstract", "paper_id")
    df["text"] = df["title"] + " " + df["text"]

    return df
