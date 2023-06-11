#################################################################################################
# This file is responsible to collect datasets and models from hugging face hub api
###############################################################################################
import pandas as pd
from huggingface_hub import HfApi, list_models, ModelFilter
from huggingface_hub import list_datasets, DatasetFilter
import logging
import sys
from dataclasses import dataclass
from huggingface_hub import snapshot_download
import os
import typing as t

@dataclass
class FilterParams:
    filter_datasets_by_tasks = ['text-classification', 'question-answering', 'summarization', 'conversational',
                                'text-generation', 'text-retrieval']
    filter_models_by_tasks = ['text-classification', 'question-answering', 'summarization', 'conversational',
                              'text-generation']


class HuggingFace:

    def __init__(self, token=None):
        self.token = token
        self.hf_api = HfApi(
            endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
            token=self.token,  # Token is not persisted on the machine.
        )
        self.models_dir = 'models'
        self.datasets_dir = 'datasets'
        self.models_path = 'hugging_face_models_info.xlsx'
        self.datasets_path = 'hugging_face_datasets_info.xlsx'

        def get_models_info(self, tasks=False, card_data=True, fetch_config=True,
                        sort="downloads") -> pd.DataFrame:
            '''
            :param tasks: list of tasks
            :param card_data: card data(dataset) information needed or not
            :param fetch_config: config information needed or not
            :param sort: sort by, default download
            :return: pandas dataframe containing models information
            '''
            models_df = pd.DataFrame()
            try:
                if tasks:
                    for task in FilterParams.filter_models_by_tasks:
                        filters = ModelFilter(task=task)
                        models = list_models(filter=filters, cardData=card_data, fetch_config=fetch_config, sort=sort,
                                            direction=-1)
                        models = [vars(x) for x in list(models)]
                        models_df = pd.concat([models_df, pd.json_normalize(models, sep='_')], axis=0)
                else:
                    models = list_models(cardData=card_data, fetch_config=fetch_config, sort=sort, direction=-1)
                    models = [vars(x) for x in list(models)]
                    models_df = pd.concat([models_df, pd.json_normalize(models, sep='_')], axis=0)
                models_df.fillna('', inplace=True)
                models_df.to_excel(f"{self.models_path}", index=False)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(exc_type, exc_tb.tb_lineno)
                logging.error("ERROR: " + str(e))
                raise e
            return models_df

    def get_models_for_a_task(self, task_list, card_data=True, fetch_config=True,
                        sort="downloads") -> pd.DataFrame:
        '''
        :param tasks: list of tasks
        :param card_data: card data(dataset) information needed or not
        :param fetch_config: config information needed or not
        :param sort: sort by, default download
        :return: pandas dataframe containing models information
        '''
        models_df = pd.DataFrame()
        try:

            for task in task_list:
                filters = ModelFilter(task=task)
                models = list_models(filter=filters, cardData=card_data, fetch_config=fetch_config, sort=sort,
                                         direction=-1)
                models = [vars(x) for x in list(models)]
                models_df = pd.concat([models_df, pd.json_normalize(models, sep='_')], axis=0)

            models_df.fillna('', inplace=True)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
            logging.error("ERROR: " + str(e))
            raise e
        return models_df

    def get_model(self, name) -> t.Dict[str, t.Any]:
        '''
        :param name: Name of the model to download 
        :return: pandas dataframe containing models information
        '''
        models_dict = {}

        model_directory = self.models_dir + "/" + name
        try:
            if not os.path.isdir(model_directory):
                os.makedirs(model_directory)
            
            filter = ModelFilter(model_name=name)
            models_dict = {}
            models = list_models(filter=filter, cardData=True, fetch_config=True)

            model_in_repo = False
            for model in models:
                print(model.modelId)
                if model.modelId == name:
                    models_dict = vars(model)
                    model_in_repo = True
                    break
               
            if model_in_repo:
                snapshot_download(repo_id=name, local_dir=model_directory, cache_dir="/nfs/hf_cache")
                logging.info("Downloaded Model" + name )
            else:
                logging.info("Warn : Model" + name + "Not in repo") 

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
            logging.error("ERROR: " + str(e))
            raise e
        return models_dict

    def get_dataset(self, name, card_data=True) -> t.Dict[str, t.Any]:
        '''
        :param dataset_filters: class huggingface_hub.DatasetFilter object
        :param size_range: filter by size of datasets,default 100M to 10B
        :param card_data: card data information
        :return: pandas dataframe
        '''

        dataset_dict = {}

        dataset_directory = self.datasets_dir + "/" + name
        try:

            if not os.path.isdir(dataset_directory):
                os.makedirs(dataset_directory)

            dataset_filter = DatasetFilter(dataset_name=name)
            datasets = list_datasets(filter=dataset_filter, cardData=card_data, full=True)
            #print(datasets)
            dataset_in_repo = False
            for d in datasets:
                if d.id == name:
                    dataset_dict = vars(d)
                    dataset_in_repo = True
                    break
            if dataset_in_repo:
                snapshot_download(repo_id=name, local_dir=dataset_directory, repo_type="dataset", cache_dir="/nfs/hf_cache")
                logging.info("Downloaded Dataset" + name )
            else:
                logging.info("Warn: Dataset" + name + "Not in repo") 
        except Exception as e:
            print(f"Trying to get dataset : {name} resulted in exception")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
            logging.error("ERROR: " + str(e))
            #raise e
        return dataset_dict


    def get_datasets_info(self, dataset_filters=True, size_range=('100M<n<1B', '1B<n<10B'),
                          card_data=True) -> pd.DataFrame:
        '''
        :param dataset_filters: class huggingface_hub.DatasetFilter object
        :param size_range: filter by size of datasets,default 100M to 10B
        :param card_data: card data information
        :return: pandas dataframe
        '''

        datasets_df = pd.DataFrame()
        try:
            if dataset_filters:
                dataset_filter = DatasetFilter()
                for task in FilterParams.filter_datasets_by_tasks:
                    for size in size_range:
                        dataset_filter.task_categories = task
                        dataset_filter.size_categories = size
                        datasets = list_datasets(filter=dataset_filter, cardData=card_data, full=True)
                        datasets = [vars(x) for x in list(datasets)]
                        datasets_df = pd.concat([datasets_df, pd.json_normalize(datasets, sep='_')], axis=0)

            else:
                datasets = list_datasets(cardData=card_data, full=True)
                datasets = [vars(x) for x in list(datasets)]
                datasets_df = pd.concat([datasets_df, pd.json_normalize(datasets, sep='_')], axis=0)
            datasets_df.fillna('', inplace=True)
            datasets_df.to_excel(f"{self.datasets_path}", index=False)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
            logging.error("ERROR: " + str(e))
            raise e
        return datasets_df

    def main(self):
        datasets_info = self.get_datasets_info()
        print(len(datasets_info))
        model_info = self.get_models_info(tasks=True)
        print(len(model_info))


if __name__ == "__main__":
    HuggingFace().main()
