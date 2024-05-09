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


@mem_cache.cache
def load_results_data(file_path):
    sys.modules['csem_gemintelligence'] = gemtelligence
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


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

# Constants for accuracy thresholds
HEAT_TREATMENT_ACCURACY = 0.999
ORIGIN_ACCURACY = 0.98

def main():
    # Load raw data for sapphire gemstones
    sapphire_data = load_raw_data("sapphire")

    # Filter data for origin and treatment tasks
    sapphire_data_val = sapphire_data["val"]
    curr_df, _ = utils.filter_basaltic_vs_basaltic(sapphire_data_val, stone_type="sapphire")
    sapphire_data["val"] = curr_df

    # Define experiment paths
    experiment_paths = {
        "XRF-UV": "multirun/2023-11-23/16-10-47/0/results"
    }
    # Process each experiment and collect results
    experiment_accuracy = {}
    experiment_stone_percentage = {}
    for experiment_type, path in experiment_paths.items():
        task_name = "pre_origin"
        task_ids = set(sapphire_data["val"].index)
        #accuracy_threshold = HEAT_TREATMENT_ACCURACY if experiment_type == "UV+FTIR" else ORIGIN_ACCURACY
        ORIGIN_ACCURACY = 0
        results = load_results_data(path)
        _, _, test_data = utils.load_results(results, available_ids=task_ids)

        # Remove from the test data the following stones
        to_ignore = {18030166,20090002,21051150,20121158,15111001,21110143,15010049,19038013,17110120,15038229,14067005,20010011,15120143,20050051,21100028,13110130}
        test_data = test_data.drop(to_ignore, level=1)
        # Create the confusion matrix for each folder
        confusion_matrices = []
        accuracies = []
        for idx, cv in enumerate(test_data.index.levels[0]):
            sub_df = test_data.loc[cv]
            
            # Keep only entries with ED and UV
            candidate_index = sub_df.index & sapphire_data["uv"].index & sapphire_data["ed"].index
            sub_df = sub_df.loc[candidate_index]
            # Create confusion matrix
            tp = (sub_df[task_name, "GT"] == "Big4") & (sub_df[task_name, "Predicted"] == "Big4")
            fp = (sub_df[task_name, "GT"] == "other") & (sub_df[task_name, "Predicted"] == "Big4")
            fn = (sub_df[task_name, "GT"] == "Big4") & (sub_df[task_name, "Predicted"] == "other")
            tn = (sub_df[task_name, "GT"] == "other") & (sub_df[task_name, "Predicted"] == "other")
            
            # Create the confusion matrix
            confusion_matrix = np.array([[sum(tp), sum(fp)], [sum(fn), sum(tn)]])
            # Calculate the accuracy
            accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / sum(sum(confusion_matrix))
            confusion_matrices.append(confusion_matrix)
            accuracies.append(accuracy)
        
        accuracy_matrix = np.array(confusion_matrices)
        accuracy = np.array(accuracies)
        
        average_confusion_matrix = np.mean(accuracy_matrix, axis=0)
        np.sum(accuracy_matrix, axis=0)
        # Calculate the accuracy
        average_accuracy_matrix = np.mean(accuracy, axis=0)
        accuracy, percent_of_stones = process_experiment(path, task_name, task_ids, ORIGIN_ACCURACY)

        experiment_accuracy[experiment_type] = accuracy
        experiment_stone_percentage[experiment_type] = percent_of_stones

        print(average_confusion_matrix)
        
    
if __name__ == "__main__":
    main()
    