from manager.load_config import CONFIG
import pickle 
import os

from tfidf.tfidf import closeDocsToTfidf

from embeddingsModels.loadModel import loadBertModelAndTokenizer, loadBioGptModelAndTokenizer
from embeddingsModels.extractEmbeddings import extractEmbeddings
from preprocessing.loadDataset import loadCloseDataset

from manager.args import readArguments

def main():
    args = readArguments()

    if args.command == "preprocess":
        if args.extractEmbeddings:
            model, tokenizer = loadBertModelAndTokenizer(args.model, args.tokenizer)
            df = loadCloseDataset(args.file)

            data = []
            for _, row in df.iterrows():
                data.append({
                    "id": row["id"],
                    "text": row["text"]
                })

            tokens_and_embeddings, embeddings = extractEmbeddings(model, tokenizer, data)

            with open(os.path.join(args.saveDir, "tokens_and_embeddings.pkl"), "wb") as file:
                pickle.dump(tokens_and_embeddings, file)
                file.close()

            with open(os.path.join(args.saveDir, "embeddings_raw.pkl"), "wb") as file:
                pickle.dump(embeddings, file)
                file.close()



    return

if __name__ == '__main__':
    main()
