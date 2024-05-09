from gemtelligence import learning
from gemtelligence.utils import load_client_data, mapper, filter_data_for_task, load_sapphires_stones_from_path
import pandas as pd 
import torch
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import omegaconf
import sys 
import gemtelligence
import pickle
import copy 
@st.cache_resource
def load_results_data(file_path):
    sys.modules['csem_gemintelligence'] = gemtelligence
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

@st.cache_resource
def load_raw_data(stone_type):
    raw_data = load_client_data(stone_type=stone_type)
    grouped_data = learning.utils.group_rechecks(raw_data, stone_type)
    return grouped_data

candidate_ids = [14020102, 19040262, 14020103, 18112031, 13120049, 16110130, 18112052, 17051193, 14060097, 13120066, 14038086, 19010119, 15060044, 14040153, 
                14080092, 15090271, 15020129, 14121059, 15040108, 14100077, 15040110, 15040111, 18111084, 19051120, 19051121, 14040183, 19051131, 17062012, 
                19110016, 19051137, 16081026, 19051143, 18100362, 19030155, 13100171, 13100172, 19110030, 16038031, 18010259, 19070102, 15040152, 19112090, 
                19070108, 18070176, 16041123, 15060136, 16080040, 14100151, 15080121, 17102010, 18042043, 14100153, 14100156, 15020223, 15100096, 19071168, 
                14010055, 18062023, 17041096, 16020171, 14030028, 14031053, 14030032, 16020180, 17038036, 16080087, 16100058, 19021020, 16100061, 17041116, 
                18041053, 17038050, 16038114, 14060260, 17097445, 17097446, 17097447, 19021032, 17097444, 17038055, 17061097, 14060268, 17061100, 19041006, 
                17097455, 14100206, 17097463, 14070007, 17097466, 18060027, 17060092, 17097467, 15030013, 16011003, 15010048, 16011008, 19042050, 18082052, 
                15010056, 15030026, 18041098, 15080211, 19062037, 14100247, 19110171, 14100261, 19062054, 17101094, 16052008, 18122025, 16100134, 19110185, 
                18038058, 16010029, 14048038, 17101104, 14110002, 17101107, 17060149, 14090040, 17101113, 15120186, 17011003, 16010043, 19038013, 15070014, 
                18102079, 14120256, 17100097, 14070083, 14070084, 16072005, 14070087, 13090119, 14070089, 18102092, 16010064, 18081111, 18081112, 19111139, 
                19080024, 14110044, 14120285, 17032029, 17032038, 19082090, 19082091, 17080170, 14120301, 17038189, 19040110, 19082096, 18080114, 14030195, 
                14070130, 18120055, 19038073, 14010235, 14100349, 15090045, 15070078, 14010245, 19100040, 16112011, 19080084, 18101141, 18120086, 18101145, 
                19100061, 14021024, 18120099, 19012006, 19012007, 18060200, 14020011, 15111084, 19101102, 14020016, 17071028, 15111093, 15111094, 14110140, 
                18060224, 18101186, 15110084, 17090002, 19012052, 17070048, 19031012, 19031013, 16111078, 14090217, 18072044, 14060012, 19081198, 19031024, 
                14020081, 15090163, 19031028, 18070011] + [17070085, 16050184, 17010193, 16050197, 18011158, 16090133, 16111129, 17010206, 13040159, 15060001,
                16050209, 18050088, 17051180, 13080115, 14100025, 17090107, 18070076, 16041021, 18031169, 13120067, 19050055, 14100042, 18097229, 17021015,
                17050200, 14120024, 14121054, 13011039, 17111139, 14100072, 15101033, 18070123, 19051122, 16040052, 18050165, 15100022, 18050169, 19031161, 
                16040061, 16081024, 17021061, 18010246, 15060101, 14061195, 15020177, 16038033, 13080211, 19031189, 18110102, 13100185, 19070106, 18100380, 
                16040093, 18070173, 18022047, 16097440, 13100193, 18110117, 16080037, 18070186, 16097451, 17110186, 16020147, 18010295, 17082040, 17060026,
                16120003, 17082052, 17020102, 15060167, 16060104, 19070151, 16038091, 13010124, 17080012, 16040142, 13010123, 17038043, 13010141, 17082078,
                17082079, 15100127, 17020131, 16038120, 19041005, 16060142, 18061044, 13010164, 17121018, 16080122, 19062011, 13010173, 17100030, 17121022, 
                14030080, 17038081, 16038145, 18082054, 17100041, 17122058, 13080337, 13050130, 19111187, 17038103, 14050072, 18040090, 19062045, 19070240, 
                13110049, 17120032, 15100195, 15020327, 17038122, 14120234, 19061035, 19020077, 15100206, 18038065, 19040049, 19021106, 14090039, 16100151, 
                19110202, 19110206, 13100357, 18102086, 18102087, 19110218, 15051084, 17075021, 16051022, 16010067, 17060181, 17060182, 14091093, 17032021, 
                18120027, 17117023, 18120033, 17120098, 18080098, 18038117, 16100200, 18020200, 16100201, 17120106, 17040234, 19040108, 19038066, 17032050, 
                16051058, 14070131, 19080054, 16030071, 13090166, 18120058, 14050179, 17072004, 17052039, 18120073, 13070223, 19100047, 18012053, 16112021, 
                16112022, 16070043, 18120105, 14030250, 17010094, 18097071, 15111089, 18120114, 17050040, 16070073, 17037245, 17092029, 19120063, 15111108, 
                19120070, 13040072, 14041034, 16010187, 18120150, 17010134, 17091030, 17090007, 17050076, 19100125, 16070123, 15050220, 18081260, 14070254, 
                13100015, 13020142, 17090036, 17090038, 17090039, 16110076, 16071166, 15090175]

def add_smoothed_peak(index, peak_height, peak_center, width):
    # Create a Gaussian bump
    bump = np.exp(-(index - peak_center)**2 / (2 * width**2))
    # Normalize the bump so that the top value is 1
    #bump /= np.max(bump)
    # Scale the bump to the desired height
    bump *= peak_height

    return bump

@st.cache_resource
def return_model(model_path):
    model_path = "results/Heat_UV+FTIR/0/0/lightning_logs/version_0/checkpoints/epoch=119-val_acc=0.99.ckpt"
    cfg = omegaconf.OmegaConf.load("results/Heat_UV+FTIR/0/.hydra/config.yaml")
    cfg.class_len = 2
    cfg.sources["uv"] = {'name': 'uv', 'index_one_row': 'keep', 'data_processing': 1, 'add_secondary_target': 0, 'input_dim': 1201} 
    cfg.sources["ftir"] = {'name': 'ftir',  'data_processing': 0, 'add_secondary_target': 0, 'input_dim': 6801} 
    cfg.secondary_class_len = 0
    model = learning.hol_net_v1.load_model(cfg, model_path)
    model = model.eval()
    return model, cfg
    

def main():
    st.title("Heat Treatment interpretability analysis")
    path = "results/Heat_UV+FTIR/0/results"
    default_sources = ["ftir", "uv"]
    class2label = mapper("treatment","sapphire", str2num=False)
    dataset_type = st.radio("Select the dataset", ["sapphire"]) #,"charateristic sapphire"])
    if dataset_type == "sapphire":
        
        sapphire_data = load_raw_data("sapphire")
        sapphire_data = copy.deepcopy(sapphire_data)
        _, pre_candidate_ids  = filter_data_for_task(sapphire_data, "treatment")
        # Get the test stons from CV 0
        results = load_results_data(path)
        entries = results.df_test.loc[0]
        test_ids = set(entries.index)
        candidate_ids = pre_candidate_ids.intersection(test_ids)
        
    else:
        sapphire_data = load_sapphires_stones_from_path("master_file_ht.h5")
        candidate_ids = set(sapphire_data["val"].index)
    # Copy the data
    
    model_path = "results/Heat_UV+FTIR/0/0/lightning_logs/version_0/checkpoints/epoch=119-val_acc=0.99.ckpt"
    # Load model 
    model, cfg = return_model( model_path)
    
    
    stone_types = st.radio("Filter test stone types", ["All", "Only heated"], index=1)
    if  stone_types == "Only heated":
        mask = sapphire_data["val"].loc[candidate_ids]["Heat Treatment Value"] == "TE"
        candidate_ids = candidate_ids.intersection(mask[mask].index)

        

    test_stone_selected = st.selectbox("Select the stone", sorted(list(candidate_ids) ))
    

    
    source = st.selectbox("Select source",  default_sources, index=0)

    aggregation_strategy = "0"
    entry = {}
    figs = {}
    
    is_augmented = True #st.checkbox("Augment the data", value=True)

    curr = sapphire_data[source].loc[test_stone_selected].copy()
    if type(curr) == pd.DataFrame and aggregation_strategy == "0":
        curr = curr.iloc[0]
        
    if is_augmented:
        if source == "ftir":
            candidates = ["3309","3186", "3232", "3311","3309", "3299","3088","3160","2730","manual"]
        elif source == "uv":
            candidates = ["800","manual"]
        peaks = st.multiselect("Select peaks to add", candidates, candidates[:2])
        # Add a smoothed peak to the curve
        columns = st.columns(len(peaks))
        for column, peak in zip(columns,peaks):
            if peak == "3309":
                default = 0.0
            else:
                default = 0.0
                
            if peak == "manual":
                peak = column.text_input("Enter the desidered peak wavelength", 0)
            else:
                # Add a placeholder
                delta = column.text_input(f"Enter the delta from {peak}", 0)
            
            modification_type = column.selectbox(f"Modification type {peak}", ["Add", "Delete"])
            if modification_type == "Add":
                peak_percentage = float(column.text_input(f"Peak percentage {peak}", default))
                
                Width = column.slider(f"Width {peak}", 1.0, 15.0, 1.0)
                

                right_peak = int(peak) + float(delta)
                if source == "uv":
                    for spectra_name in ["A", "B"]:
                        if right_peak not in curr[spectra_name].index:
                            if spectra_name == "A":
                                st.error(f"Peak {right_peak} not in the UV spectrum. Ignoring it")
                                continue
                            else:
                                continue
                        peak_height = curr[spectra_name].loc[right_peak] * (peak_percentage / 100 )
                        to_be_added = add_smoothed_peak(curr[spectra_name].index, peak_height, int(peak) + float(delta), Width)
                        curr[spectra_name] += to_be_added
                
                elif source == "ftir":
                    if right_peak not in curr.index:
                        st.error(f"Peak {right_peak} not in the FTIR spectrum. Ignoring it")
                        continue
                    peak_height = curr.loc[right_peak] * (peak_percentage / 100 )
                
                    to_be_added = add_smoothed_peak(curr.index, peak_height, int(peak) + float(delta), Width)
                    curr += to_be_added
            elif modification_type == "Delete":
                delta = column.text_input(f"Delta {peak}", 0)
                width = column.text_input(f"Width {peak}", 0)
                starting_index = int(peak) - float(delta) - float(width)
                ending_index = int(peak) + float(delta) + float(width)
                
                # Create a line that is connect starting and ending index, and it is also tangent to the curve
                starting_value = curr.loc[starting_index]
                ending_value = curr.loc[ending_index]
                
                slope = (ending_value - starting_value) / (ending_index - starting_index)
                
                indices = np.arange(starting_index, ending_index+1)
                y = slope * (indices - starting_index) + starting_value
                
                
                
                curr.loc[starting_index:ending_index] =y

            
            fig = go.Figure()
            if source == "uv":
                for spectra_name in ["A", "B"]:
                    fig.add_trace(go.Scatter(x=curr[spectra_name].index, y=curr[spectra_name].values, mode='lines'))
                fig.update_layout(title="", xaxis_title="Wavelength", yaxis_title="Intensity")
                
                # Swap the x
                fig.update_layout(xaxis=dict(autorange="reversed"))
            elif source == "ftir":
                fig.add_trace(go.Scatter(x=curr.index, y=curr.values, mode='lines'))
                fig.update_layout(title="", xaxis_title="Wavelength", yaxis_title="Intensity")
                
                # Swap the x
                fig.update_layout(xaxis=dict(autorange="reversed"))
            else:
                raise ValueError("Unknown source")
    st.plotly_chart(fig)

    curr_copy = curr.copy()
        
    if source == "uv":
        if type(curr) == pd.DataFrame:
            chosen = st.selectbox("Which measurement do you want to visualize for UV?", list(range(len(curr))))
            curr = curr.iloc[chosen]
        
        channel_a  = curr["A"]
        channel_b = curr["B"]
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

    
    if not "uv" in entry and "uv" in default_sources:
        uv_values = np.array([0.0] * 2 * 1201)[None,:]
        entry["uv"] = uv_values
    
    if not "ftir" in entry and "ftir" in default_sources:
        ftir_values = np.array([0.0] * 6801)[None,:]
        entry["ftir"] = ftir_values
    

    
    task = "Treatment"
    if task == "Treatment":
        entry["uv"] = pd.DataFrame(entry["uv"], index=[test_stone_selected], columns=[f"uv_{idx}" for idx in range(entry["uv"].shape[1])])
        entry["ftir"] = pd.DataFrame(entry["ftir"], index=[test_stone_selected], columns=[f"ftir_{idx}" for idx in range(entry["ftir"].shape[1])])    
    
   
    
        
    entry = learning.data.final_validation_processing(entry, entry.keys())
    
    for entry_source in entry.keys():
        entry[entry_source] = torch.tensor(entry[entry_source]).float()

    
  
        
        
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
    
    
    ## ONLY FOR THE PAPER
    if "ftir" == source and True:
        indices = np.where((curr.index >= 3200) & (curr.index <= 3400))[0]
        values = curr_copy.iloc[indices]

        x_axis = curr_copy.index[indices]
        # Create a matplotlib figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        # Add a dashed vertical line at 3309 and 3232 cm-1
        ax.axvline(3309, linestyle="--", color="black")
        ax.axvline(3232, linestyle="--", color="black")
        
        ax.plot(x_axis, values)
        font = 15
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        
        plt.legend(fontsize=font/1.3 )

        ax.set_xlabel("Wavelength", fontsize=font)
        ax.set_ylabel("Intensity", fontsize=font)
        # Flip the x axis
        ax.locator_params(axis='x', nbins=9) 
        ax.invert_xaxis()
        
        # Set the y axis with a delta of 0.5 starting from the minimum value
        min_value = values.min()
        max_value = values.max()
        ax.set_ylim(min_value - 0.05, min_value + 0.6)
        
        col1_spectrum.pyplot(fig)
        
        # Save in SVG 
        import datetime 
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        prediction = class2label(top_pred)
        save_pic = st.button("Save the spectrum")
        if save_pic:
            fig.savefig(f"/home/treatedgem/repos/6298bb88f671f40f1bd763b3/figs/spectrum_{date}_pred_id_{test_stone_selected}_{prediction}_{is_augmented}.png", format="png", dpi=500)
            
        
        # Do the same thing for indices between 2600 and 2800
        indices = np.where((curr.index >= 2600) & (curr.index <= 2800))[0]
        values = curr_copy.iloc[indices]
        x_axis = curr_copy.index[indices]
        # Create a matplotlib figure
        fig, ax = plt.subplots()
        ax.plot(x_axis, values)
        
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        
        plt.legend(fontsize=font/1.3 )

        ax.set_xlabel("Wavelength", fontsize=font)
        ax.set_ylabel("Intensity", fontsize=font)
        ax.locator_params(axis='x', nbins=9) 
        # Flip the x axis
        ax.invert_xaxis()
        col2_spectrum.pyplot(fig)
    

        





if __name__=="__main__":
    main()