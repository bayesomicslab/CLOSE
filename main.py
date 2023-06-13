from manager.load_config import CONFIG
import pandas as pd
from tfidf.tfidf import closeDocsToTfidf

def main():
    # ADD CODE HERE
    data_df = pd.read_csv("all_files_with_abstract_titles.csv")

    closeDocsToTfidf(data_df, 500, "all_files_tfidf.csv")

    return

if __name__ == '__main__':
    main()
