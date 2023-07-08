from typing import Callable, Dict, List, Tuple

import gzip
from transformers import PreTrainedModel
import glob
import torch
import gc
import psutil
import os
import subprocess
import numpy as np
import h5py
from pympler import muppy
from pympler import summary

__base_ram_usage = 0
__batch_ram_usage = -1

class __BatchEmbeddingData:
    def __init__(self):
        self.ids = []
        self.embeddings = []

    def clearRam(self):
        for id in self.ids:
          del id
        del self.ids

        for embedding in self.embeddings:
          del embedding
        del self.embeddings

        self.ids = []
        self.embeddings = []


def __unloadRam(data: __BatchEmbeddingData, batch_num: int, save_dir: str=".", ram_use_limit_percentage: int = 60):
    """
    Save the batch of embeddings passed in into the save_dir if ram usage is being breached

    :param embeddings: the embeddings to be saved
    :param batch_num: the batch number this embeddings was produced from 
    :param save_dir: the directory to save to
    """

    global __base_ram_usage
    global __batch_ram_usage

    if __batch_ram_usage == -1:
        __batch_ram_usage = psutil.virtual_memory()[2] - __base_ram_usage

    print(f"BASE RAM: {__base_ram_usage}\tBATCH_RAM:{__batch_ram_usage}\tCUR_RAM:{psutil.virtual_memory()[2]}")

    if psutil.virtual_memory()[2] + __batch_ram_usage > ram_use_limit_percentage:
        all_objects = muppy.get_objects()
        sum = summary.summarize(all_objects)
        summary.print_(sum)
        data.clearRam()

        return True

    return False

def __processLoad(data: __BatchEmbeddingData, batch_ids: List, batch_tokenized: Dict, batch_num: int, zip_num: int, model: PreTrainedModel, run: Callable, device: str = "cpu", save_dir: str="."):
    """
    process the corresponding load on the model and extract embeddings


    :param id_embeddings: the ids of the embedding to save to, passed in by reference to make faster
    :param extracted_embeddings: the extracted embedding to save to, passed in by reference to make faster
    :param batch_id: the ids of the batch
    :param batch_tokenized: the tokenized text of the batch
    :param mode: the model to run with
    :param run: the function to be run 
    :param device: the device to send the tensors to
    """

    print(f"MOVING BATCH {batch_num} TO GPU")
    tokenized_batch_to_device = {
        "input_ids": torch.stack(batch_tokenized["input_ids"]).to(device),
        "token_type_ids": torch.stack(batch_tokenized["token_type_ids"]).to(device),
        "attention_mask": torch.stack(batch_tokenized["attention_mask"]).to(device)
    }
    print(f"FINISHED MOVING BATCH {batch_num} TO GPU")
    
    print(f"RUNNING BATCH {batch_num} EXTRACT EMBEDDINGS")
    
    embeddings_batch = run(model, (batch_ids, tokenized_batch_to_device))
    print(f"FINISHED RUNNING BATCH {batch_num} EXTRACT EMBEDDINGS")
    
    print(f"MOVING BATCH {batch_num} TO CPU")
    
    with open(os.path.join(save_dir, f"ids_{zip_num}.txt"), "a") as idfile:
      for id in embeddings_batch[0]:
        idfile.write(str(id)+"\n")
        
    with open(os.path.join(save_dir, f"embeds_{zip_num}.csv"), "a") as embfile:
      cls_embeddings = embeddings_batch[1].detach().cpu().numpy()[:,0,:]
      np.savetxt(embfile,cls_embeddings,delimiter=',')

    print(f"extracted_embeddings: len {len(data.embeddings) if data.embeddings is not None else 'None'}")
    print(f"FINISHED MOVING BATCH {batch_num} TO CPU")
    
def __memoryCleanup():
    print(f"COLLECTING GARBAGE")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"FINISHED COLLECTING GARBAGE")

def __cleanFiles(save_dir):
    filelist = glob.glob(os.path.join(save_dir, "ids_*.txt"))
    filelist.extend(glob.glob(os.path.join(save_dir, "embeds_*.csv")))
    filelist.extend(glob.glob(os.path.join(save_dir, "ids_*.txt.gz")))
    filelist.extend(glob.glob(os.path.join(save_dir, "embeds_*.csv.gz")))
    for f in filelist:
        os.remove(f)

def extractEmbeddingsLoadSplit(data: Tuple[List, Dict], model: PreTrainedModel, run: Callable, split_size: int = 1000, save_dir: str = "."):
    """
    Split the load on tokenized data to be run on a function save to disk if ram is being overloaded

    :param data: the data to be run with load splitting
    :param model: the model to be passed into runnable
    :param run: the function to be called with model and dat for the extraction of embeddings :param split_size: the size of the splits to be used
    :param save_dir: the directory to save to when ram is being overflowed
    """
    global __base_ram_usage
    global __batch_ram_usage

    os.makedirs(save_dir, exist_ok=True)
    # clean up bad runs
    __cleanFiles(save_dir)
    
    torch.cuda.empty_cache()

    with torch.no_grad():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
        model = model.to(device)
    
        batch_embedding_data = __BatchEmbeddingData()
    
        id_batch = []
        tokenized_batch = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        cnt = 0
        cur_batch = 0
        ram_batch = 0
        zip_num = 0
        total_extracted = 0

        __base_ram_usage = psutil.virtual_memory()[2]
        
        for id, input_ids, token_type_ids, attention_mask in zip(data[0], data[1]["input_ids"], data[1]["token_type_ids"], data[1]["attention_mask"]):
            if cnt == split_size:
                print(cur_batch)
                print("-"*50)
                __processLoad(batch_embedding_data, id_batch, tokenized_batch, cur_batch, zip_num, model, run, device, save_dir=save_dir)
                print(len(id_batch))
                print(total_extracted)
                total_extracted += len(id_batch)

                id_batch = []
                tokenized_batch = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
                cnt = 0
                cur_batch += 1

                if __unloadRam(batch_embedding_data, ram_batch, save_dir=save_dir):
                    ram_batch += 1

                    __memoryCleanup()

                    all_objects = muppy.get_objects()
                    sum = summary.summarize(all_objects)
                    summary.print_(sum)

                else:
                    __memoryCleanup()


                print("-"*50)
    
            id_batch.append(id)
            tokenized_batch["input_ids"].append(input_ids)
            tokenized_batch["token_type_ids"].append(token_type_ids)
            tokenized_batch["attention_mask"].append(attention_mask)
            cnt += 1

            if total_extracted >= 30000:
                print("ZIPPING THE SAVED EMBEDDINGS")
                zip_cmd = f"gzip \"{os.path.join(save_dir, f'ids_{zip_num}')}.txt\" ; gzip \"{os.path.join(save_dir, f'embeds_{zip_num}.csv')}\""
                !{zip_cmd}
                print("FINISHED ZIPPING THE SAVED EMBEDDINGS")
                zip_num+=1
                total_extracted=0

        if cnt > 0:
            print("-"*50)
            __processLoad(batch_embedding_data, id_batch, tokenized_batch, cur_batch, zip_num, model, run, device, save_dir=save_dir)
            total_extracted += len(batch_embedding_data[0])
            id_batch = []
            tokenized_batch = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
            cnt = 0
            cur_batch += 1

            __batch_ram_usage = 100 # WARNING: Stupid trick to force ram unloader to save files, don't change but don't do this trick either

            if __unloadRam(batch_embedding_data, ram_batch, save_dir=save_dir):
                ram_batch += 1

            __memoryCleanup()

            print("-"*50)

