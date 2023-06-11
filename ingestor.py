#################################################################################################
# This file is responsible to collect datasets and models from hugging face hub api
###############################################################################################

from hugging_face_api import HuggingFace
from cmf_ingest import cmf_ingest
import argparse
from cmflib import cmfquery
from unidecode import unidecode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model to download")
    parser.add_argument("task", type=str)
    args = parser.parse_args()
    hf = HuggingFace()
    cmf_query = cmfquery.CmfQuery("mlmd")
    artifact_list = cmf_query.get_all_artifacts()
    model_df = hf.get_models_for_a_task([args.task], True, False)
    len_model_df = len(model_df)
    print("===============================")    
    print(f" length of model dataset {len_model_df}")

    print("=================")
    print(model_df["modelId"])
    print("=================")
    #exit()
    for index, row in model_df.iterrows():
        try:
            license_model = row["cardData_license"]
            if license_model:
                model = row["modelId"]
                print("========================================================")
                print(f"*******************Getting {model}*********************")
                res = [i for i in artifact_list if model in i]
                if len(res) == 0:
                    model_dict = hf.get_model(model)

                    cmf_Ingest = cmf_ingest(args.task, model)
                    cmf_Ingest.add_model(model_dict)

                    cardData = model_dict.get('cardData', None)
                    datasets = None
                    if datasets:
                        for d in datasets:
                            data_dict = hf.get_dataset(d)
                            if len(data_dict) != 0:
                                dataset_new_dict = {}
                                for k, v in data_dict.items():
                                    if (k) == "description":
                                        continue

                                    dataset_new_dict[k] = unidecode(str(v)).replace('"', "'")
                                d_name = dataset_new_dict["id"]

                                print("========================================================")
                                print(f"*******************Getting {d_name}*********************")

                                cmf_Ingest.add_dataset(dataset_new_dict)
                    cmf_Ingest.cmf.finalize()
                    print(f"Ingeseted to cmf {model}")
                else:
                    print(f" model {model} already ingested")
            else:
                print(f" model {model} do not have license")
        except Exception as e:
            print(f"Error while ingesting {model} {e}")     
        

