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
import matplotlib.pyplot as plt


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
def load_results_data(file_path):
    sys.modules['csem_gemintelligence'] = gemtelligence
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def main():
    raw_df = utils.load_client_data(stone_type="sapphire")
    filtered_df_origin = utils.return_df_for_paper(raw_df, "origin", filter_noisy=False)
    available_ids_origin = filtered_df_origin["val"].index
    filtered_df_heat = utils.return_df_for_paper(raw_df, "treatment", filter_noisy=False)
    available_ids_heat = filtered_df_heat["val"].index

    # Load results (Models are exactly the same as results/Origin_UV+XRF/0/results, and results/Heat_UV+FTIR/0/results. But the evaluation is done on the consistency dataset.
    # Experiments Paths
    
    experiment_paths = {
    "UV+XRF": "results/Origin_Consistency_UV+XRF/results", 
    "UV+FTIR": "results/Heat_Consistency_UV+FTIR/results"
    }
    
    
    for is_what, path in enumerate(experiment_paths.values()):
        for_plot = None
        for threhsold_val in [0,0.98,0.99]:
            res = load_results_data(path)
            to_check = res.df_test
            
            # We
            if is_what == 0: # Origin 
                available_ids = available_ids_origin
                uv_ed_origin_path = "results/Origin_UV+XRF/0/results"
                tmp_thr = load_results_data(uv_ed_origin_path)
    
            elif is_what == 1: # Treatment.  
                available_ids = available_ids_heat
                uv_ftir_heat_path = "results/Heat_UV+FTIR/0/results"
                tmp_thr = load_results_data(uv_ftir_heat_path)

            confidence_value_9 =  utils.return_threshold(tmp_thr, accuracy=threhsold_val, available_ids=available_ids)
            if threhsold_val == 0:
                
                confidence_value_9 = [0 for x in confidence_value_9]

            # Drop level 0 of index 
            dropped_to_check = to_check#.droplevel(0)
            
            tmp = utils.load_client_data("sapphire")
            clusters_rechecks = {}
            for entry, row in tmp["val"].iterrows():
                clusters_rechecks[entry] = row["Rechecks"]
            
            # Create a set of unique rechecks
            fin_group = set()
            for key, group in clusters_rechecks.items():
                tuple_group = tuple(group)
                if tuple_group not in fin_group:
                    fin_group.add(tuple_group)
            
            # Keep only groups longer than 1
            fin_group = [x for x in fin_group if len(x) > 1]
            #fin_group2 = [x for x in fin_group if len(x) > 2]
            
            fin_list = []
            for idx, group in enumerate(fin_group):
                for entry in group:
                    if entry in dropped_to_check.index.levels[1]:
                        dropped_to_check.loc[dropped_to_check.index.droplevel(0) == entry, "group"] = idx
                        
            if is_what == 0:
                to_check = ("origin","Predicted")
                to_match = ("origin","Matched")
                gt = ("origin","GT")
                thr = 0
            else:
                to_check = ("treatment","Predicted") 
                to_match = ("treatment","Matched")  
                gt = ("treatment","GT")
                thr = 0

            # Group by group
            consistency = []
            dropped_to_check["confidence"] = dropped_to_check["Pred"].max(axis=1)
            fin_group_df = []
            res = []
            cnt = 0
            acc = []
            test = []
            for idx, group in dropped_to_check.groupby("group"):
                # if is_what == 1:
                #     breakpoint()
                thr_idx = np.unique(group.index.droplevel(1))[0]    
                threshold = confidence_value_9[thr_idx]
                #threshold = 0
                if is_what == 0:
                    if idx in [112,116,226,240,514,564,677,810]:
                        continue
                elif is_what == 1:
                    if idx in [564,677]:
                        continue
                # breakpoint()
                group = group.loc[group["confidence"] > threshold]
                
                # Ignore stones that are corrupted
                if 14121063 in group.index.droplevel(0) or  15080114 in group.index.droplevel(0) or 15080067 in group.index.droplevel(0) or 15080068 in group.index.droplevel(0) or (17100020 in group.index.droplevel(0)):
                    continue
                                    
                if (len(group) > 1 and (1,19082102) not in group.index) or for_plot is not None:
                    test.append(group)
                    is_matched = any(group[to_match] == True)
                    acc.append(is_matched)
                    
                    if group[to_check].nunique() == 1:
                        consistency.append(1)
                        
                    else:
                        consistency.append(0)
                    fin_group_df.append(group)    
                    
                    for ii, name in enumerate(group.index.droplevel(0)):
                        
                        # Move the stone to mid-month to improve the visualization
                        yymm = utils.extract_date(name)
                        if name == 18102079:
                            import datetime
                            yymm = datetime.date(2018, 11, 15)
                            
                        if name == 16038113:
                            import datetime
                            yymm = datetime.date(2016, 4, 15)
                            
                        if name == 16038086:
                            import datetime
                            yymm = datetime.date(2016, 4, 15)
                            
                        if name == 16037137:
                            import datetime
                            yymm = datetime.date(2016, 4, 15)
                            
                        pred = group.iloc[ii][to_check]
                        curr_gt = group.iloc[ii][gt]
                        res.append([name, yymm, cnt, pred, curr_gt ])
                        
                    cnt += 1

            if for_plot is None:
                for_plot = pd.DataFrame(res, columns=["stone_id", "yymm", "group", "pred", "gt"])
        
                import seaborn as sns
                # Order by group    
                yymm_for_plot = for_plot.sort_values("yymm")
                yymm_for_plot["order_yymm"] = yymm_for_plot["yymm"].rank(method="dense")
                # Change the order of the groups based on the order_yymm
                res = []
                yymm_for_plot["new_group"] = np.nan
                for _, group in yymm_for_plot.groupby("group"):
                    order_min = group["order_yymm"].min()
                    group["new_group"] = order_min
                    res.append(group)
                
                
                curr = pd.concat(res).sort_values("new_group")
                # Make sure new_group is contiguous (no gaps)
                curr["new_group"] = curr["new_group"].rank(method="dense")
                
                curr = curr.sort_values("yymm")
                to_add = 0
                dfs = []
                for _, group in curr.groupby("new_group"):
                    group["new_group"] = group["new_group"] + to_add
                    if group["group"].nunique() > 1:
                        for ii, jj in enumerate(group.groupby("group")):
                            idx, inner_new_group = jj
                            if ii > 0:
                                inner_new_group["new_group"] = inner_new_group["new_group"] + ii
                            dfs.append(inner_new_group)
                        to_add += ii             
                        
                    else:
                        dfs.append(group)            
            
                for_plot = pd.concat(dfs)
                for_plot["group"] = -for_plot["new_group"]
                
            else:
                novel = pd.DataFrame(res, columns=["stone_id", "yymm", "group", "pred", "gt"])
                # Set eveything to nan that is not in the new df
                for_plot.loc[~for_plot["stone_id"].isin(novel["stone_id"]),"pred"] = np.nan

            palette = sns.color_palette()
            palette[0] = "black"
            if is_what == 0:
                color_dict = {"Sri Lanka": palette[0], "Madagascar": palette[0], "Burma": palette[0], "Kashmir": palette[0]}
            elif is_what == 1:
                color_dict = {"TE": palette[0], "NTE": palette[0]}
            
            # Add a is_consisten column if in the group there is more than one prediction
            for_plot["is_consistent"] = 0
            for idx, group in for_plot.groupby("group"):
                # Iterate over the group and remove entries that are NaN
                
                
                
                if group["pred"].nunique() > 1:
                    for_plot.loc[for_plot["group"] == idx, "is_consistent"] = 1
                    
                for _, row in group.iterrows():
                    
                    if not type(row["pred"]) == str and np.isnan(row["pred"]):
                        
                        # Make it label 2
                        for_plot.loc[for_plot["stone_id"] == row["stone_id"], "is_consistent"] = 2
                
                
            color_dict = {0: "black", 1: "red", 2: (1, 1, 1, 0)}
            size_dict = [15 if x != 2 else 0 for x in for_plot["is_consistent"]]
            
            
            fig = sns.scatterplot(data=for_plot.iloc[::-1], x="yymm", y="group", hue="is_consistent", s=size_dict,  zorder=4, palette=color_dict)

                
                
            # Add an horizontal line for each group it has to be in the background. 
            # The line should connect the first and last point of the group
            # if for_plot_new is None:
            #     for_plot_new = for_plot
            for idx, group in for_plot.groupby("group"):
                # Sort values by stone_id
                group = group.sort_values("stone_id")
                
                if group["pred"].nunique() > 1:
                    # The lines shoule be in the background (i.e. behind the points)
                    plt.plot([group["yymm"].min(), group["yymm"].max()], [idx, idx], color="red",  linewidth=0.6, zorder=1)
          
                else:
                    # check that the group are not all nan:
                    #if group["pred"].isna().sum() != len(group["pred"]):

                    plt.plot([group["yymm"].min(), group["yymm"].max()], [idx, idx], color="black",linewidth=0.4, zorder=1)

            # Set the y axis to be between the min and max group
            plt.ylim(for_plot["group"].min()-1.5, for_plot["group"].max()+1.5)

            # Make sure the lines are behind the points
            plt.gca().set_axisbelow(True)
            
        
   
            # Remove the y ticks
            plt.yticks([])
            plt.xlabel("Date")
            plt.ylabel("Stone ID")
            
            # Number of stones considered 
            print("Percentage of stones considered: ", sum(for_plot["pred"].isna() == False)/len(for_plot))
            
            font = 15
            # Remove legend's title
            plt.gca().get_legend().set_title(None)
            # Increase legend font size
            plt.gca().legend(fontsize=font)
            # Increase xticks font size
            plt.xticks(fontsize=font)
            # Increase yticks font size
            plt.yticks(fontsize=font)
            # Increase xlabel font size
            plt.xlabel("Date", fontsize=font)
            plt.ylabel("Stone ID", fontsize=font)
            # Increase ylabel font size
            
            # Remove legend
            plt.gca().legend().remove()

            plt.savefig(f"plots/longitudinal_{is_what}_{threhsold_val}.png", dpi=600)
            pd.concat(fin_group_df).to_csv(f"to_check_{is_what}.csv")
            plt.clf()

if __name__ == "__main__":
    main()
