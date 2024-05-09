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


# # Helper functions
# def filter_data_for_task(raw_data, task):
#     filtered_data = utils.return_df_for_paper(raw_data, task, filter_noisy=True)
#     available_ids = filtered_data["val"].index
#     return filtered_data, available_ids

# def get_subconclusion(filtered_data, available_ids, task):
#     subconclusion, one_pred = utils.get_subconclusion_matrix(
#         filtered_data["val"], filter_ids=available_ids, subconclusion_type=task
#     )
#     return subconclusion, one_pred

# def remove_disagreeing_stones(one_pred, available_ids):
#     disagreeing_indices = one_pred.loc[(one_pred["icp"] == False) | (one_pred["micro"] == False)].index
#     return [x for x in available_ids if x not in disagreeing_indices]

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
    origin_data, origin_ids = utils.filter_data_for_task(sapphire_data, "origin")
    treatment_data, treatment_ids = utils.filter_data_for_task(sapphire_data, "treatment")

    # Load expert results for origin and treatment tasks
    origin_expert_results, origin_expert_predictions = utils.get_subconclusion(origin_data, origin_ids, "origin")
    treatment_expert_results, treatment_expert_predictions = utils.get_subconclusion(treatment_data, treatment_ids, "treatment")

    # Remove stones for origin determination where experts's ICP and Micro do not agree
    origin_ids = utils.remove_disagreeing_stones(origin_expert_predictions, origin_ids)

    # Define experiment paths
    experiment_paths = {
        "UV+FTIR": "results/Heat_UV+FTIR/0/results",
        "XRF": "results/Origin_XRF/0/results",
        "UV": "results/Origin_UV/0/results",
        "UV+XRF": "results/Origin_UV+XRF/0/results", #"results/Origin_UV+XRF/0/results"
    }

    # Process each experiment and collect results
    experiment_accuracy = {}
    experiment_stone_percentage = {}
    for experiment_type, path in experiment_paths.items():
        task_name = "treatment" if experiment_type == "UV+FTIR" else "origin"
        task_ids = treatment_ids if experiment_type == "UV+FTIR" else origin_ids
        accuracy_threshold = HEAT_TREATMENT_ACCURACY if experiment_type == "UV+FTIR" else ORIGIN_ACCURACY

        accuracy, percent_of_stones = process_experiment(path, task_name, task_ids, accuracy_threshold)

        experiment_accuracy[experiment_type] = accuracy
        experiment_stone_percentage[experiment_type] = percent_of_stones

    # Extract results for the expert and Gemtelligence
    expert_xrf_acc_origin = origin_expert_results.loc[1, "xrf_acc"]
    expert_xrf_num_stones_origin = origin_expert_results.loc[1, "xrf_rep"]
    expert_uv_acc_origin = origin_expert_results.loc[1, "uv_acc"]
    expert_uv_num_stones_origin = origin_expert_results.loc[1, "uv_rep"]
    expert_uv_xrf_acc_origin = origin_expert_results.loc[1, "uv+xrf_acc"]
    expert_uv_xrf_num_stones_origin = origin_expert_results.loc[1, "uv+xrf_rep"]
    expert_uv_ftir_acc_treatment = treatment_expert_results.loc[1, "uv+ftir_acc"]
    expert_uv_ftir_num_stones_treatment = treatment_expert_results.loc[1, "uv+ftir_rep"]

    gem_uv_xrf_acc_origin = experiment_accuracy["UV+XRF"]
    gem_uv_xrf_num_stones_origin = experiment_stone_percentage["UV+XRF"]
    gem_uv_acc_origin = experiment_accuracy["UV"]
    gem_uv_num_stones_origin = experiment_stone_percentage["UV"]
    gem_xrf_acc_origin = experiment_accuracy["XRF"]
    gem_xrf_num_stones_origin = experiment_stone_percentage["XRF"]
    gem_uv_ftir_acc_treatment = experiment_accuracy["UV+FTIR"]
    gem_uv_ftir_num_stones_treatment = experiment_stone_percentage["UV+FTIR"]
    
    
    data =[
        ["Gemtelligence", "calibration 2", gem_uv_xrf_num_stones_origin, gem_uv_xrf_acc_origin, "uv+xrf"],
        ["Human",  "", expert_uv_num_stones_origin, expert_uv_acc_origin, "uv"],
        ["Human",  "", expert_xrf_num_stones_origin, expert_xrf_acc_origin, "xrf"],
        ["Human",  "", expert_uv_xrf_num_stones_origin, expert_uv_xrf_acc_origin, "uv+xrf"],
        ["Human",  "", expert_uv_ftir_num_stones_treatment, expert_uv_ftir_acc_treatment, "uv+ftir"],
        ["Gemtelligence",  "calibration 2", gem_uv_num_stones_origin, gem_uv_acc_origin, "uv"],
        ["Gemtelligence",  "calibration 2", gem_xrf_num_stones_origin, gem_xrf_acc_origin, "xrf"],  
        ["Gemtelligence", "calibration 2", gem_uv_ftir_num_stones_treatment, gem_uv_ftir_acc_treatment, "uv+ftir"],
      
    ]
    
    
    df = pd.DataFrame(data, columns=["Model", "calibration", "Stones", "Accuracy", "Method"])
    df["Stones"] = df["Stones"] * 100
    df["Accuracy"] = df["Accuracy"] * 100
    
    # Create a scatter plot with x = "Stones" and y = "Accuracy"
    sns.scatterplot(x="Stones", y="Accuracy", data=df, hue="Method", style="Model", s=100)
    # Get the color palette
    palette = sns.color_palette()
    
    # Create an arrows that goes from 0.89,0.12  to 0.93,0.45
    for stone_type in ["uv", "xrf", "uv+xrf", "uv+ftir"]:
        if stone_type == "xrf":
            shift_x= 0.015
            shift_y = 0.0025
            df_curr = df[df["Method"] == "xrf"]
            cc = palette[2]
            # Arrow should be dashed
            
        elif stone_type == "uv":
            shift_x= 0.015
            shift_y = 0.0015
            df_curr = df[df["Method"] == "uv"]
            cc = palette[1]

        elif stone_type == "uv+xrf":
            shift_x= 0.015
            shift_y = 0.001
            df_curr = df[df["Method"] == "uv+xrf"]
            cc = palette[0]
        elif stone_type == "uv+ftir":
            shift_x= 0.015
            shift_y = 0.0001
            df_curr = df[df["Method"] == "uv+ftir"]
            cc = palette[3]
            

        x_init = df_curr[df_curr["Model"] == "Human"]["Stones"].values[0]
        y_init = df_curr[df_curr["Model"] == "Human"]["Accuracy"].values[0]
        x_end = df_curr[df_curr["calibration"] == "calibration 2"]["Stones"].values[0]
        y_end = df_curr[df_curr["calibration"] == "calibration 2"]["Accuracy"].values[0]
        x_start = x_init + shift_x
        y_start = y_init + shift_y
        goal_x = x_end - shift_x
        goal_y = y_end - shift_y
        
        arrowprops = {
                "arrowstyle": "->",
                "linewidth": 1,
                "color": cc,
                "linestyle": "--",
            }
        plt.annotate("", xy=(goal_x, goal_y), xytext=(x_start, y_start) , arrowprops=arrowprops) 
        
    # Change the x axis in stones considered/stones above threshold
    font = 15
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    
    plt.legend(fontsize=font/1.3 )
    # Make the legend on the top centre with 2 columns
    #plt.legend(loc="upper center", ncol=2, fontsize=font)
    plt.xlabel("Confidently classified stones [%]", fontsize=font)
    plt.ylabel("Accuracy [%]", fontsize=font)
    
    # Make in the legend Method and Model in bold
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label in ["Method", "Model"]:
            label = f"$\\bf{{{label}}}$"
        else:
            # Capitalize all the letters if not Gemtelligence or Human
            if not (label == "Human" or label == "Gemtelligence"):
                label = label.upper()
        new_labels.append(label)
    
    plt.legend(handles, new_labels, fontsize=font/1.3)
    
   
    # If the plots folder does not exist, create it
    Path("plots").mkdir(parents=True, exist_ok=True)
    
    # Save in the tmps folder
    plt.savefig("plots/figure_2.png", bbox_inches="tight")
    
if __name__ == "__main__":
    main()
    