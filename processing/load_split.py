from typing import Callable, Dict, List, Tuple

from dataclasses import dataclass, field
from transformers import PreTrainedModel
import glob
import torch
import gc
import psutil
import os
import h5py
import zipfile

__BASE_RAM_USAGE = 0
__BATCH_RAM_USAGE = -1

@dataclass
class __BatchEmbeddingData:
    ids: List = field(default_factory=lambda:[])
    embeddings: List = field(default_factory=lambda:[])

    def clear(self):
        del self.ids
        del self.embeddings

        self.ids = []
        self.embeddings = []

@dataclass
class __BatchTokenizedData:
    ids: List = field(default_factory=lambda:[])
    tokenized: Dict[str, List] = field(default_factory=lambda:{"input_ids": [], "token_type_ids": [], "attention_mask": []})

    def clear(self):
        del self.ids
        del self.tokenized

        self.ids = []
        self.tokenized = {"input_ids": [], "token_type_ids": [], "attention_mask": []}

@dataclass
class __DataFiles:
    files: List = field(default_factory=lambda:[])

    def clear(self):
        del self.files
        self.files = []


def __unloadRam(data: __BatchEmbeddingData, embeddings_data_files: __DataFiles, batch_num: int, save_dir: str=".", ram_use_limit_percentage: int = 60):
    """
    Save the batch of embeddings passed in into the save_dir if ram usage is being breached

    :param embeddings: the embeddings to be saved
    :param batch_num: the batch number this embeddings was produced from 
    :param save_dir: the directory to save to
    """

    global __BASE_RAM_USAGE
    global __BATCH_RAM_USAGE

    if __BATCH_RAM_USAGE == -1:
        __BATCH_RAM_USAGE = psutil.virtual_memory()[2] - __BASE_RAM_USAGE

    print(f"BASE RAM: {__BASE_RAM_USAGE}\tBATCH_RAM:{__BATCH_RAM_USAGE}\tCUR_RAM:{psutil.virtual_memory()[2]}")

    if psutil.virtual_memory()[2] + __BATCH_RAM_USAGE > ram_use_limit_percentage:
        save_path: str = os.path.join(save_dir, f"embeddings_num_{batch_num}.h5")
        embeddings_data_files.files.append(save_path)
        with h5py.File(save_path, "w") as hf:
            hf.create_dataset("ids", data.ids)
            hf.create_dataset("embeddings", data.embeddings)
            hf.close()
        return True

    return False

def __processLoad(tokenized_data: __BatchTokenizedData, embeddings_data: __BatchEmbeddingData, batch_num: int, model: PreTrainedModel, run: Callable, device: str = "cpu"):
    """
    process the corresponding load on the model and extract embeddings

    :param data: the extracted embedding to save to, passed in by reference to make faster
    :param batch_num: the number of the batch being run
    :param model: the model to run with
    :param run: the function to be run 
    :param device: the device to send the tensors to
    """

    print(f"MOVING BATCH {batch_num} TO GPU")
    tokenized_batch_to_device = {
        "input_ids": torch.stack(tokenized_data.tokenized["input_ids"]).to(device),
        "token_type_ids": torch.stack(tokenized_data.tokenized["token_type_ids"]).to(device),
        "attention_mask": torch.stack(tokenized_data.tokenized["attention_mask"]).to(device)
    }
    print(f"FINISHED MOVING BATCH {batch_num} TO GPU")

    
    print(f"RUNNING BATCH {batch_num} EXTRACT EMBEDDINGS")
    embeddings_batch = run(model, (tokenized_data.ids, tokenized_batch_to_device))
    print(f"FINISHED RUNNING BATCH {batch_num} EXTRACT EMBEDDINGS")

    del tokenized_batch_to_device

    print(f"MOVING BATCH {batch_num} TO CPU")
    for id, cls_embeddings in zip(embeddings_batch[0], embeddings_batch[1].cpu().numpy()):
        embeddings_data.ids.append(id)
        embeddings_data.embeddings.append(cls_embeddings)
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
    filelist.extend(glob.glob(os.path.join(save_dir, "embeddings_num_*.h5")))
    filelist.extend(glob.glob(os.path.join(save_dir, "embeddings_zip_batch_*.zip")))

    for f in filelist:
        os.remove(f)

def __compressFiles(files: __DataFiles, filename: str):
    """
    Compress the files in the file list so that local memory won't be filled

    :param files: the list of files i need to compress
    """

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zipf:
        for f in files.files:
            zipf.write(f)

    for f in files.files:
        os.remove(f)


def extractEmbeddingsLoadSplit(data: Tuple[List, Dict], model: PreTrainedModel, run: Callable, split_size: int = 1000, zip_size: int = 1000, save_dir: str = "."):
    """
    Split the load on tokenized data to be run on a function save to disk if ram is being overloaded

    :param data: the data to be run with load splitting
    :param model: the model to be passed into runnable
    :param run: the function to be called with model and dat for the extraction of embeddings :param split_size: the size of the splits to be used
    :param save_dir: the directory to save to when ram is being overflowed
    """
    global __BASE_RAM_USAGE
    global __BATCH_RAM_USAGE

    os.makedirs(save_dir, exist_ok=True)
    # clean up bad runs
    __cleanFiles(save_dir)
    
    torch.cuda.empty_cache()

    with torch.no_grad(): # No detach needed with this
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    
        batch_embedding_data = __BatchEmbeddingData()
        batch_tokenized_data = __BatchTokenizedData()
        embeddings_data_files = __DataFiles()
    
        cnt = 0
        ram_cnt = 0

        __BASE_RAM_USAGE = psutil.virtual_memory()[2]
        
        for id, input_ids, token_type_ids, attention_mask in zip(data[0], data[1]["input_ids"], data[1]["token_type_ids"], data[1]["attention_mask"]):
            if cnt % split_size == 0 and cnt: # If the split size has been reached the process the batch through model in gpu and return results
                print("-"*50)
                print(f"BATCH: {cnt // split_size}")
                __processLoad(batch_tokenized_data, batch_embedding_data, cnt//split_size, model, run, device=device)
                batch_tokenized_data.clear()

                if __unloadRam(batch_embedding_data, embeddings_data_files, ram_cnt, save_dir=save_dir): # If ram has been used up then unload the RAM in order to continue saving new embeddings
                    batch_embedding_data.clear()
                    ram_cnt += 1
                
                __memoryCleanup()
                print("-"*50)
    
            batch_tokenized_data.ids.append(id)
            batch_tokenized_data.tokenized["input_ids"].append(input_ids)
            batch_tokenized_data.tokenized["token_type_ids"].append(token_type_ids)
            batch_tokenized_data.tokenized["attention_mask"].append(attention_mask)
            cnt += 1

            if ram_cnt%zip_size == 0 and ram_cnt:
                __compressFiles(embeddings_data_files, os.path.join(save_dir, f"embeddings_zip_batch_{ram_cnt//zip_size}.zip"))
                embeddings_data_files.clear()
                __memoryCleanup()

        if cnt > 0:
            print("-"*50)
            print(f"BATCH: {cnt // split_size}")
            __processLoad(batch_tokenized_data, batch_embedding_data, cnt//split_size, model, run, device=device)
            batch_tokenized_data.clear()

            __BATCH_RAM_USAGE = 100 # WARNING: Stupid trick to force ram unloader to save files, don't change but don't do this trick either
            if __unloadRam(batch_embedding_data, embeddings_data_files, ram_cnt, save_dir=save_dir): # If ram has been used up then unload the RAM in order to continue saving new embeddings
                batch_embedding_data.clear()
                ram_cnt += 1

            __memoryCleanup()
            print("-"*50)


        if ram_cnt%zip_size != 0 and ram_cnt:
            __compressFiles(embeddings_data_files, os.path.join(save_dir, f"embeddings_zip_batch_{(ram_cnt//zip_size) + 1}.zip"))
            embeddings_data_files.clear()
            __memoryCleanup()

