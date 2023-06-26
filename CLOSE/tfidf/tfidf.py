# base libraries
from typing import List
import re
import os

# data sci libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# local libraries
from manager.load_config import LOCAL


def __normalizeDataset(df: pd.DataFrame):
    """
    Normalizes all sentences in dataset to remove all of the unwanted characters for tfidf

    :param df: the dataframe that needs to be normalized
    """

    pat = re.compile(r"[^a-zA-Z ]+")

    for i, row in df.iterrows():
        n_text = row["text"]
        n_text = re.sub(pat, "", n_text)
        df.at[i, "text"] = n_text.lower()

    return df

def tfIdf(docs: List[str]):
    """
    Tf-Idf vectorize the list of documents

    :param docs: list of strings where each string is a doc
    :return: a tuple of the sparse matrix and the feature names
    return = (sparse_matrix, feature_names_list)
    """
    tfidf = TfidfVectorizer()
    doc_term_matrix = tfidf.fit_transform(docs) # Learn vocab and idf from training set. Transform returns sparse matrix of the result

    return (doc_term_matrix.todense(), np.array(tfidf.get_feature_names_out()))

def mapTfIdfToDocs(doc_term_matrix: np.ndarray, feature_names: np.ndarray, doc_list: List[str], max_length: int):
    """
    Map the TfIdf sparse matrix to the docs in word order

    :param doc_term_matrix: the matrix representing the transformed documents in tfidf
    :param feature_names: the list of feature names from the tfidf
    :param doc_list: the list of documents to map
    :param max_length: the maximum length to transform into tfidf

    :return: the new list of documents mapped by tfidf in order
    """

    new_doc_list = []

    # Iterate over the doc term matrix per doc array
    for i, doc_term_array in enumerate(doc_term_matrix):
        # Organize the features by the tfidf weight and extract the top max_length of them for use
        doc_features = pd.DataFrame(doc_term_array.T, index=feature_names, columns=["tfidf_score"]).sort_values("tfidf_score", ascending=False)
        assert doc_features is not None

        max_length = max(max_length, len(doc_features.index))
        doc_features = doc_features.head(max_length)

        # Iterate over the broken up document and make a new document only using the words with the highest tf-idf
        new_doc = ""
        for word in doc_list[i].split(" "):
            if word not in doc_features.index:
                continue
            new_doc += word + " "

        new_doc_list.append(new_doc[:-1])

    return new_doc_list


def closeDocsToTfidf(df: pd.DataFrame, max_length: int, doc_name: str):
    """
    Run TF-Idf vectorization on the dataframe

    :param df: the dataframe that contains the tfidf to run against under the "text" column
    :param max_length: the maximum length to transform the tfidf into
    :return:
    """

    df["text"] = df["title"] + "\n" + df["asbtract"]

    df = __normalizeDataset(df)
    text = df["text"].tolist()
    docs_matrix, feature_names = tfIdf(text)
    res_text = mapTfIdfToDocs(docs_matrix, feature_names, text, max_length)
    df["tfidf_text"] = res_text

    df.to_csv(os.path.join(LOCAL, "data", "tmp", doc_name), sep='\t')

    return df


# run if run directly
if __name__ == "__main__":
    # Test this crap
    t = [
        "I enjoy reading about Machine Learning, and Machine Learning is my PhD subject.",
        "I would enjoy, a walk in the park.",
        "I was reading, in the library."
    ]

    tmp_df = pd.DataFrame(t, columns=["text"])


    tmp_df = __normalizeDataset(tmp_df)
    t = tmp_df["text"].tolist()
    docs_matrix, feature_names = tfIdf(t)
    res = mapTfIdfToDocs(docs_matrix, feature_names, t, 5)
    tmp_df["text"] = res

    print(tmp_df)
