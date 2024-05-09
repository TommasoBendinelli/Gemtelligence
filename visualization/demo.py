import streamlit as st 
from gemtelligence import learning
import omegaconf
from gemtelligence import utils, learning
from joblib import Memory
import gemtelligence 
import sys
import pickle
import plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import torch
sys.modules['csem_gemintelligence'] = gemtelligence

cache_dir = './cachedir'
mem_cache = Memory(cache_dir, verbose=0)

@mem_cache.cache
def load_results_data(file_path):
    sys.modules['csem_gemintelligence'] = gemtelligence
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

@mem_cache.cache
def load_raw_data(stone_type):
    raw_data = utils.load_client_data(stone_type=stone_type)
    grouped_data = learning.utils.group_rechecks(raw_data, stone_type)
    return grouped_data



def main():
    # Set page config to wide
    st.set_page_config(layout="wide")
    st.title("GemIntelligence Demo")
    st.write("This is a demo of GemIntelligence")
    
    # Load raw data for sapphire gemstones
    stone_type = st.selectbox("Select a stone type", ["Client Stones"], index =0)
    sapphire_client_data = load_raw_data("sapphire")
    if stone_type == "Raw Client Stones" or stone_type == "Client Stones":
        path = "data/master_file.h5"
        
    if stone_type == "Client Stones":
        sapphire_data = sapphire_client_data
    
    aggregation_strategy = st.selectbox("Select an aggregation strategy", ["Mean", "0"])

    
    task = st.selectbox("Select a task", ["Origin", "Treatment"])
    
    
    if task == "Origin":
        model_path = "results/Origin_UV+XRF/0/0/lightning_logs/version_0/checkpoints/epoch=164-val_acc=0.90.ckpt"
        cfg = omegaconf.OmegaConf.load("results/Origin_UV+XRF/0/.hydra/config.yaml")
        cfg.class_len = 4
        cfg.sources["ed"] = {'name': 'ed', 'cancel_out_threshold': True, 'num_columns': 26, 'val_method': 'max', 'is_log': False, 'is_z_normalized': False}  
        cfg.sources["icp"] = {'name': 'icp', 'include_strongly_mislabelled_data': '???', 'include_additional_info': True, 'include_additional_info2': False, 'cancel_out_threshold': True, 'include_ED': False, 'data_feeding': 'single', 'calibrate': False, 'include_reference_data': -1, 'unsupervised_idea': False, 'remove_features': 'None', 'val_method': 'max', 'num_columns': 16, 'is_log': False}
        cfg.sources["uv"] = {'name': 'uv', 'index_one_row': 'keep', 'data_processing': 1, 'add_secondary_target': 0, 'input_dim': 1201} 
        cfg.secondary_class_len = 0
        path = "results/Origin_UV+XRF/0/results"
        all_sources = ["xrf", "icp", "uv"]
        default_sources = ["xrf", "uv"]
        class2label = utils.mapper("origin","sapphire", str2num=False)
        # Filter data for origin and treatment tasks
        if stone_type == "Client Stones":
            _, pre_candidate_ids = utils.filter_data_for_task(sapphire_data, "origin")

        
    
    elif task == "Treatment":
        model_path = "results/Heat_UV+FTIR/0/0/lightning_logs/version_0/checkpoints/epoch=119-val_acc=0.99.ckpt"
        cfg = omegaconf.OmegaConf.load("results/Heat_UV+FTIR/0/.hydra/config.yaml")
        cfg.class_len = 2
        cfg.sources["uv"] = {'name': 'uv', 'index_one_row': 'keep', 'data_processing': 1, 'add_secondary_target': 0, 'input_dim': 1201} 
        cfg.sources["ftir"] = {'name': 'ftir',  'data_processing': 0, 'add_secondary_target': 0, 'input_dim': 6801} 
        cfg.secondary_class_len = 0
        
        path = "results/Heat_UV+FTIR/0/results"
        all_sources = ["ftir", "uv"]
        default_sources = ["ftir", "uv"]
        class2label = utils.mapper("treatment","sapphire", str2num=False)
        _, pre_candidate_ids  = utils.filter_data_for_task(sapphire_data, "treatment")
        

    # Load model 
    model = learning.hol_net_v1.load_model(cfg, model_path)
    model = model.eval()

   
   
    if "Client Stones" == stone_type:
        # Get the test stons from CV 0
        results = load_results_data(path)
        entries = results.df_test.loc[0]
        test_ids = set(entries.index)
        candidate_ids = pre_candidate_ids.intersection(test_ids)

    if task == "Origin":
        all_ed_data = sapphire_data["ed"].loc[candidate_ids]["Concentration"]
    
        # Remove non-sense measurements from ICP
        for col in sapphire_data["icp"].columns:
            sapphire_data["icp"][col] =  sapphire_data["icp"][col].apply(pd.to_numeric,errors="coerce")
            sapphire_data["icp"] = sapphire_data["icp"].loc[~(sapphire_data["icp"][col] <0)]

        sapphire_data["icp"] = sapphire_data["icp"].loc[~(sapphire_data["icp"]["Al27"] < 500000)]
        sapphire_data["icp"] = sapphire_data["icp"].loc[~(sapphire_data["icp"]["Nd146"] > 10)]
    
        # For ICP we have NaNs, we replace them with values that are still in the range of data but cannot be confused with the real values
        all_icp_data = sapphire_data["icp"].loc[candidate_ids]
        for col in all_icp_data.columns:
            tmp =  sapphire_data["icp"][col]
            min_val = tmp.min()
            mean_val = tmp.mean()
            diff = (mean_val - min_val)*0.5
            to_be_replaced = min(0,min_val - diff)
            # Fill the NaN values with the minimum value
            all_icp_data[col].fillna(to_be_replaced, inplace=True)
        

    test_stone_selected = st.selectbox("Select a stone", sorted(list(candidate_ids)))
    

    # Get the corresponding data
    
    sources = st.multiselect("Select Sources for the Network", all_sources, default_sources )

    
    entry = {}
    figs = {}
    for source in sources:
        if source == "xrf":
            curr = all_ed_data.loc[test_stone_selected]   
            if type(curr) == pd.DataFrame and aggregation_strategy == "0":
                curr = curr.iloc[-1]
        elif source == "icp":
            curr = all_icp_data.loc[test_stone_selected]
            if type(curr) == pd.DataFrame and aggregation_strategy == "0":
                curr = curr.iloc[0]
        elif source == "uv":
            curr = sapphire_data[source].loc[test_stone_selected]
            if type(curr) == pd.DataFrame and aggregation_strategy == "0":
                # Consider the one for which A and B are different
                candidate = curr.loc[(curr["A"] != curr["B"]).any(axis=1)]
                if candidate.shape[0] > 0:
                    curr = candidate.iloc[0]
                else:
                    curr = curr.iloc[0]
                
        elif source == "ftir":
            curr = sapphire_data[source].loc[test_stone_selected]
            if type(curr) == pd.DataFrame and aggregation_strategy == "0":
                curr = curr.iloc[0]
        
        # Create a parallel plot where each axis is a chemical element
        dimensions = []
        if source == "xrf":
            if type(curr) == pd.DataFrame:
                chosen = st.selectbox("Which measurement do you want to visualize for ED?", list(range(len(curr))))
                curr = curr.iloc[chosen]
                
            for element in curr.index:
                if all_ed_data[element].min() != all_ed_data[element].max():
                    dimensions.append(dict(range = [all_ed_data[element].min(),all_ed_data[element].max()],
                                        label = element, values = [curr[element]]))
            fig = go.Figure(data=go.Parcoords(
                                line_color='blue',
                                dimensions = dimensions))
            # Make the plot fill the entire width
            fig.update_layout(width=600, height=300)
            
            # Make sure the last axis is not cut off
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
            
            figs[source] = fig
            ed_values =  np.array(curr.values)
            entry["ed"] = ed_values[None,:]
            
            
            
        elif source == "icp":
            if type(curr) == pd.DataFrame:
                chosen = st.selectbox("Which measurement do you want to visualize for ICP?", list(range(len(curr))))
                curr = curr.iloc[chosen]
                
            for element in curr.index:
                dimensions.append(dict(range = [all_icp_data[element].min(),all_icp_data[element].max()],
                                        label = element, values = [curr[element]]))
            fig = go.Figure(data=go.Parcoords(
                                line_color='blue',
                                dimensions = dimensions))
            
            fig.update_layout(width=600, height=300)
            
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
            
            figs[source] = fig 
            
            
            
            icp_values =  np.array(curr.values)
            entry["icp"] = icp_values[None,:]
            
            
        elif source == "uv":
            if type(curr) == pd.DataFrame:
                chosen = st.selectbox("Which measurement do you want to visualize for UV?", list(range(len(curr))))
                curr = curr.iloc[chosen]
                
                
            # Plot the spectrum
            fig = go.Figure()

            channel_a  = curr["A"]
            channel_b = curr["B"]
            
            fig.add_trace(go.Scatter(x=channel_a.index, y=channel_a.values, mode='lines', name="Channel A"))
            fig.add_trace(go.Scatter(x=channel_b.index, y=channel_b.values, mode='lines', name="Channel B"))
            fig.update_layout(title="", xaxis_title="Wavelength", yaxis_title="Intensity")
            
            # Remove padding on the top
            fig.update_layout(margin=dict(l=50, r=50, t=0, b=50))
            fig.update_layout(width=600, height=300)
            
            figs[source] = fig 
            
            
            uv = np.concatenate([channel_a.values, channel_b.values])[None,:]
            entry["uv"] = uv
        
        elif source == "ftir":
            # Plot the spectrum
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curr.index, y=curr.values, mode='lines'))
            fig.update_layout(title="", xaxis_title="Wavelength", yaxis_title="Intensity")
            
            # Remove padding on the top
            fig.update_layout(margin=dict(l=50, r=50, t=0, b=50))
            fig.update_layout(width=600, height=300)
            
            figs[source] = fig
       
            ftir = curr.values[None,:]
            entry["ftir"] = ftir
            
            
        
            
        
    # Do the inference
    if not "ed" in entry and "xrf" in all_sources:
        ed_values = np.array([0.0] *  len(all_ed_data.columns))[None,:]
        entry["ed"] = ed_values
    
    if not  "icp" in entry and "icp" in all_sources:
        icp_values = np.array([0.0] *  len(all_icp_data.columns))[None,:]
        entry["icp"] = icp_values
        
    if not "uv" in entry and "uv" in all_sources:
        uv_values = np.array([0.0] * 2 * 1201)[None,:]
        entry["uv"] = uv_values
        
    if not "ftir" in entry and "ftir" in default_sources:
        ftir_values = np.array([0.0] * 6801)[None,:]
        entry["ftir"] = ftir_values
    
    
    if task == "Origin":
        entry["ed"] = pd.DataFrame(entry["ed"], index=[test_stone_selected], columns=[f"ed_{idx}" for idx in range(entry["ed"].shape[1])])
        entry["icp"] = pd.DataFrame(entry["icp"], index=[test_stone_selected], columns=[f"icp_{idx}" for idx in range(entry["icp"].shape[1])] )
        entry["uv"] = pd.DataFrame(entry["uv"], index=[test_stone_selected], columns=[f"uv_{idx}" for idx in range(entry["uv"].shape[1])])
    elif task == "Treatment":
        entry["uv"] = pd.DataFrame(entry["uv"], index=[test_stone_selected], columns=[f"uv_{idx}" for idx in range(entry["uv"].shape[1])])
        entry["ftir"] = pd.DataFrame(entry["ftir"], index=[test_stone_selected], columns=[f"ftir_{idx}" for idx in range(entry["ftir"].shape[1])])    
    
    entry = learning.data.final_validation_processing(entry, entry.keys())
    
    for source in entry.keys():
        entry[source] = torch.tensor(entry[source]).float()

    output = model.forward(entry)
    probs = torch.nn.Softmax(dim=1)(output)
    # Get the top prediction and its probability (confidence)
    top_prob, top_pred = torch.max(probs, dim=1)
    top_prob = float(top_prob)
    top_pred = int(top_pred)
    
    
    
    # Write the prediction
    st.markdown(f"**Prediction**: {class2label(top_pred)}")
    st.markdown(f"**Confidence**: {top_prob}")

    # Get the ground truth
    if task == "Origin":
        st.markdown(f"**Ground truth**: {sapphire_data['val'].loc[test_stone_selected]['FinalOrigin']}")
    elif task == "Treatment":
        st.markdown(f"**Ground truth**: {sapphire_data['val'].loc[test_stone_selected]['Heat Treatment Value']}")
    else:
        raise ValueError("Unknown task")
    
        
    
    col1_tabular, col2_tabular = st.columns(2)
    col1_spectrum, col2_spectrum = st.columns(2)
    if "xrf" in figs:
        col1_tabular.markdown(f"**Source**: XRF")
        col1_tabular.plotly_chart(figs["xrf"])
    
    if "icp" in figs:
        col2_tabular.markdown(f"**Source**: ICP")
        col2_tabular.plotly_chart(figs["icp"])
    
    if "uv" in figs:
        col1_spectrum.markdown(f"**Source**: UV")
        col1_spectrum.plotly_chart(figs["uv"])
    
    if "ftir" in figs:
        col2_spectrum.markdown(f"**Source**: FTIR")
        col2_spectrum.plotly_chart(figs["ftir"])
        
    
    

    
    
    
if __name__ == "__main__":
    main()