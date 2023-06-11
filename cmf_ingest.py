#################################################################################################
# This file is responsible to collect datasets and models from hugging face hub api
###############################################################################################
import pandas as pd
import re
from huggingface_hub import HfApi, list_models, ModelFilter
from huggingface_hub import list_datasets, DatasetFilter
import logging
import sys
from dataclasses import dataclass
from cmflib import cmf
import typing as t
import string
from hugging_face_api import HuggingFace
import argparse
import os
import shutil
from unidecode import unidecode
class cmf_ingest:


    def __init__(self, task, model_name):
        self.task = task
        self.model_name = model_name
        self.cmf = cmf.Cmf("mlmd", task, graph=True)
        self.cmf.create_context(model_name)
        self.cmf.create_execution(model_name)
        self.model_dir = "models"
        self.dataset_dir = "datasets"


    def add_model(self, model_info) -> bool:
        #name: str = model_info.get("name", "test")
        name = model_info["modelId"]
        print("printing name")
        print(name)
        path = os.path.join(self.model_dir, name)
        
        #print(:model_info["modelId"][0])
        if not name:
            print("name is space")
            return False
        else:
            model_new_dict = {}
            for k, v in model_info.items():
                model_new_dict[k] = unidecode(str(v)).replace('"', "'")
            self.cmf.log_model(path=path, event="OUTPUT", custom_properties=model_new_dict)
            print("ingested")


    def add_dataset(self, dataset_info) -> bool:
        #name: str = model_info.get("name", "test")
        name = dataset_info["id"]
       # print("printing name")
       # print(name)
        path = os.path.join(self.dataset_dir, name)
        
        #print(:model_info["modelId"][0])
        if not name:
            print("Warn : Name information not available")
            return False
        else:
            print("executing cmf")
            #print("===============Ann===============")
            #print(dataset_info)
            dataset_new_dict = {}
            for k, v in dataset_info.items():
                dataset_new_dict[k] = str(v)
            self.cmf.log_dataset(url=path, event="INPUT", custom_properties=dataset_new_dict)
            print("ingested")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model to download")
    parser.add_argument("task", type=str)
    parser.add_argument("model", type=str)
    args = parser.parse_args()
    hf = HuggingFace()
    model_dict = hf.get_model(args.model)
    if len(model_dict) != 0:

        cmf_Ingest = cmf_ingest(args.task, args.model)
        cmf_Ingest.add_model(model_dict)

        cardData = model_dict.get('cardData', None)
        datasets = cardData.get('datasets', None) if cardData.get('datasets', None) else None
        if datasets:
            for d in datasets:
                data_dict = hf.get_dataset(d)
                if len(data_dict) != 0:
                    dataset_new_dict = {}
                    for k, v in data_dict.items():
                        if (k) == "description":
                            #print(v)
                            continue

                        dataset_new_dict[k] = str(v)
                    #print(dataset_new_dict)
                    cmf_Ingest.add_dataset(dataset_new_dict)
    cmf_Ingest.cmf.finalize()
    print("Ingeseted to cmf")
