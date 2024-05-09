import pandas as pd
import seaborn as sns
import numpy as np 
from collections import defaultdict
import gemtelligence
from pathlib import Path
import sys
from gemtelligence import utils, learning
sys.modules['csem_gemintelligence'] = gemtelligence
import pickle 
from joblib import Memory


sys.modules['csem_gemintelligence'] = gemtelligence

cache_dir = './cachedir'
mem_cache = Memory(cache_dir, verbose=0)


palette = {
        'UV+XRF+ICP': sns.color_palette()[4],
        'ICP': sns.color_palette()[5],
        'UV+XRF': sns.color_palette()[0],
        'UV': sns.color_palette()[1],
        'XRF': sns.color_palette()[2],
        "UV+FTIR": sns.color_palette()[7],
        "FTIR": sns.color_palette()[9],
    }

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

def third_plot_origin(entry_meta, sources, palette, target):
    
    plot_entry_meta = {x: entry_meta[x] for x in sources}

    ys = {key: entry["y"] for key, entry in plot_entry_meta.items()}
    
    df = pd.DataFrame(ys, index=entry_meta["UV"]["x"])*100

    df.index = df.index*100

    df[r"% Number of stones"] = df.index
    dfs = []
    for key in sources:
        tmp = df[[r"% Number of stones",key]].dropna()
        # Rename column key to Accuracy
        tmp = tmp.rename(columns={key: "Accuracy"})
        tmp["source"] = key
        if "_s" in key:
            tmp["style"] = "only"
        else:
            tmp["style"] = "all"
        dfs.append(tmp)
    df = pd.concat(dfs,axis=0)
    # Reset index
    df = df.reset_index(drop=True)
    
    df = pd.concat(dfs,axis=0)
    # Reset index
    df = df.reset_index(drop=True)

    df[r"Normalized Confidence [%]"] = df[r"% Number of stones"].values[::-1]
    
    # Do a rolling average with a window of 3 for Accuracy
    for source in sources:
        tmp = df.loc[df["source"] == source, "Accuracy"].rolling(3, center=False, min_periods=0, closed="left").mean()
        tmp.iloc[:1] = df.loc[df["source"] == source, "Accuracy"].iloc[:1]
        # Fillnan with last value 
        df.loc[df["source"] == source, "Accuracy"] = tmp
        
    g = sns.lineplot(
        data=df,
        x=r"% Number of stones", hue="source",
        y="Accuracy",
        linewidth=1.5,
        palette=palette,
        sort=False,
    )
    
    # Remove legend title
    g.legend_.set_title(None)
    
    g.set_ylabel("Accuracy %")
    
    # Set x label
    g.set_xlabel("Stones above threshold [%]")
    
    # Limit x axis between 0 and 1
    g.set_xlim(0,100)
    
    fig = g.get_figure()

    # Invert x axis
    fig.gca().invert_xaxis()
        
    fig.savefig(f"plots/figure_3_{target}.png", bbox_inches="tight", dpi=1200)
    
    fig.clf()
    

   
        
def preprocessing_acc_stone_percent(df, target, stone_type, is_it_new=True):
    key_order = utils.target_list(target.lower(),stone_type)

    cv_x = {}
    cv_y = {}
    
    for cv in df.index.levels[0]:
        accuracy_dict = {}
        threshold_dict = {}
        
        curr_df = df.loc[cv]
        curr_df = curr_df.sort_values("confidence", ascending=False)
        for i in reversed(range(len(curr_df))):
            if i == 0:
                break
            curr = curr_df.iloc[:i]

            gt = curr[(target,"Predicted")]            
            accuracy = curr[(target,"Matched")].sum()/len(curr)

            threhsold = curr["confidence"].iloc[-1]
            
            entry =  ((i/len(curr_df))// 0.01) * 0.01
            accuracy_dict[entry] = accuracy
            threshold_dict[entry] = threhsold
        
        x = sorted(list(accuracy_dict.keys()),reverse=True)
        y = [accuracy_dict[stone] for stone in x]    
        cv_x[cv] = x
        cv_y[cv] = y
    # Take the mean of the cross validation
    ys = np.array([[cv_y[cv]] for cv in cv_y.keys()]).squeeze()
    y = np.mean(ys, axis=0)
    y_var = np.std(ys, axis=0)
    y = np.mean([cv_y[cv] for cv in cv_y.keys()], axis=0)
    x = np.mean([cv_x[cv] for cv in cv_x.keys()], axis=0)
    x = [np.round(x,2) for x in x]
    # Make sure that all the keys are multiplier of 0.0015 
    
    return x, y, y_var



def prepare_for_data_plot(path, available_ids, target="origin"):
    results = load_results_data(path)
    _, _, test_data = utils.load_results(results, available_ids=available_ids)
    x,y, y_var = preprocessing_acc_stone_percent(test_data, target, "sapphire") 

    return x, y, y_var 



TASK = "origin" # You can choose between "origin" and "treatment"
sources_origin = ["UV+XRF+ICP","ICP","UV+XRF", "UV","XRF"] #ALL is UV+XRF+ICP
sources_heat = ["FTIR","UV+FTIR","UV"]
def main():
    # Load raw data for sapphire gemstones
    sapphire_data = load_raw_data("sapphire")
    
    # Filter data for origin and treatment tasks
    origin_data, origin_ids = utils.filter_data_for_task(sapphire_data, "origin")
    treatment_data, treatment_ids = utils.filter_data_for_task(sapphire_data, "treatment")

    # Load expert results for origin and treatment tasks
    origin_expert_results, origin_expert_predictions = utils.get_subconclusion(origin_data, origin_ids, "origin")

    # Remove stones for origin determination where experts's ICP and Micro do not agree
    origin_ids = utils.remove_disagreeing_stones(origin_expert_predictions, origin_ids)
    
    # Define experiment paths
    if TASK == "origin":
        experiment_paths = {
            "XRF": "results/Origin_XRF/0/results", #"runs/sapphire_origin_hol_net/2023-05-08/13-35-12/0/results", #"results/Origin_XRF/0/results",
            "UV": "results/Origin_UV/0/results", #"results/Origin_UV/0/results", #"runs/sapphire_origin_hol_net/2023-05-08/13-35-12/0/results", #"results/Origin_UV/0/results",
            "UV+XRF": "results/Origin_UV+XRF/0/results",
            "ICP": "results/Origin_ICP/0/results",
            "UV+XRF+ICP": "results/Origin_UV+XRF+ICP/0/results",
        }
        available_ids = origin_ids
        sources = sources_origin
    elif TASK == "treatment":
        experiment_paths = {
            "UV+FTIR": "results/Heat_UV+FTIR/0/results",
            "UV": "results/Heat_UV/0/results",
            "FTIR": "results/Heat_FTIR/0/results"
        }
        available_ids = treatment_ids
        sources = sources_heat
    else:
        raise ValueError("TASK must be either 'origin' or 'treatment'")
    
    to_plot_data = {}
    for source in sources:
        x, y, y_var = prepare_for_data_plot(experiment_paths[source], available_ids,TASK)
        to_plot_data[source] = {"x": x, "y": y, "y_var": y_var}
    
    third_plot_origin(to_plot_data, sources, palette, target=TASK)
    
    
if __name__ == "__main__":
    main()