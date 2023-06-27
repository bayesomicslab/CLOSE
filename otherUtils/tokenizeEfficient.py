from ..embeddingsModels import *
from ..preprocessing import *
import os
import pickle

print("LOADING DATASET")
#df = loadCloseDataset("./all_files_with_abstract_titles.zip")
df = loadCloseDataset("./small_files_with_abstract_titles.csv")
print("FINISHED LOADING DATASET")

print("LOADING MODEL AND TOKENIZER")
model, tokenizer = loadBertModelAndTokenizer("bert-base-uncased", "bert-base-uncased")
print("LOADING MODEL AND TOKENIZER")

print("PREPROCESSING FOR TPKENIZATION")
ids = []
data_str = []
for _, row in df.iterrows():
  ids.append(row["id"])
  data_str.append(row["text"])
print("FINISHED PREPROCESSING FOR TOKENIZATION")

print("STARTING TOKENIZATION")
tokenized_text = tokenizer(data_str, truncation=True, padding="max_length",  return_tensors="pt")
print("FINISHED TOKENIZATION")
print(tokenized_text)

print("POSTPROCESSING TOKENIZATION")
tokenized_with_ids = (id, tokenized_text)
print("FINISHED POSTPROCESSING TOKENIZATION")

print("SAVING TOKENIZATION")
with open(os.path.join(".", "tokenized_text.pkl"), "wb") as file:
  pickle.dump(tokenized_with_ids, file)
  file.close()
print("FINISHED SAVING TOKENIZATION")
