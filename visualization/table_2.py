import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from gemtelligence import utils, learning
import pickle
from joblib import Memory
import gemtelligence
from pathlib import Path
import sys

sys.modules['csem_gemintelligence'] = gemtelligence

cache_dir = './cachedir'
mem_cache = Memory(cache_dir, verbose=0)


@mem_cache.cache
def load_raw_data(stone_type):
    raw_data = utils.load_client_data(stone_type=stone_type)
    grouped_data = learning.utils.group_rechecks(raw_data, stone_type)
    return grouped_data

def get_subconclusion(filtered_data, available_ids, task):
    subconclusion, one_pred = utils.get_subconclusion_matrix(
        filtered_data["val"], filter_ids=available_ids, subconclusion_type=task
    )
    return subconclusion, one_pred


@mem_cache.cache
def load_results_data(file_path):
    sys.modules['csem_gemintelligence'] = gemtelligence
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

# Helper functions
def filter_data_for_task(raw_data, task):
    filtered_data = utils.return_df_for_paper(raw_data, task, filter_noisy=True)
    available_ids = filtered_data["val"].index
    return filtered_data, available_ids

def remove_disagreeing_stones(one_pred, available_ids):
    disagreeing_indices = one_pred.loc[(one_pred["icp"] == False) | (one_pred["micro"] == False)].index
    return [x for x in available_ids if x not in disagreeing_indices]

def process_experiment(path, task_name, task_ids, accuracy_threshold):
    results = load_results_data(path)
    _, _, test_data = utils.load_results(results, available_ids=task_ids)
    threshold = utils.return_threshold(results, accuracy=accuracy_threshold, available_ids=task_ids)
    stones_above_threshold = [test_data.loc[cv].loc[test_data.loc[cv]["confidence"] >= float(threshold[idx])]
                              for idx, cv in enumerate(test_data.index.levels[0])]
    
    final_df = pd.concat(stones_above_threshold)
    accuracy = sum(final_df[task_name, "Matched"]) / len(final_df)
    percent_of_stones = len(final_df) / len(test_data)
    
    return accuracy, percent_of_stones


def main():
    # Load raw data for sapphire gemstones
    sapphire_data = load_raw_data("sapphire")

    # Filter data for origin and treatment tasks
    origin_data, origin_ids = filter_data_for_task(sapphire_data, "origin")
    treatment_data, treatment_ids = filter_data_for_task(sapphire_data, "treatment")

    # Load expert results for origin and treatment tasks
    origin_expert_results, origin_expert_predictions = get_subconclusion(origin_data, origin_ids, "origin")

    # Remove stones for origin determination where experts's ICP and Micro do not agree
    origin_ids = remove_disagreeing_stones(origin_expert_predictions, origin_ids)

    # Define experiment paths
    experiment_paths = {
        "UV+FTIR": "results/Heat_UV+FTIR/0/results",
        "UV+XRF": "results/Origin_UV+XRF/0/results"
    }
    
    # Define accuracy thresholds
    NO_MODE = 0
    MODE_1 = 0.98
    MODE_2 = 0.99
    
    # Process each experiment and collect results
    
    res = {}
    for mode in [NO_MODE, MODE_1, MODE_2]:
        
        experiment_accuracy = {}
        experiment_stone_percentage = {}
        for experiment_type, path in experiment_paths.items():
            task_name = "treatment" if experiment_type == "UV+FTIR" else "origin"
            task_ids = treatment_ids if experiment_type == "UV+FTIR" else origin_ids            
            accuracy_threshold = mode
            accuracy, percent_of_stones = process_experiment(path, task_name, task_ids, accuracy_threshold)
            experiment_accuracy[experiment_type] = accuracy
            experiment_stone_percentage[experiment_type] = percent_of_stones

        tmp = {
            "accuracy": experiment_accuracy,
            "stone_percentage": experiment_stone_percentage
        }

        res[mode] = pd.DataFrame(tmp)
    
    out = pd.concat(res, axis=1)
    print(out)
    # Save results in plot folder into a csv file
    out.to_csv("table_2.csv")
    
if __name__ == "__main__":
    main()