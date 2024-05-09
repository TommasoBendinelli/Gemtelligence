from omegaconf import OmegaConf
from gemtelligence import utils
import gemtelligence
from gemtelligence.learning import hol_net_v1
import pandas as pd
import pickle
import torch
from torch.nn.functional import softmax
import io

#deprecated hardcoded test-function
def get_data(stone_id, stone_type, is_recheck):
    if is_recheck:
        raise NotImplementedError
    analytical_data = utils.load_client_data(stone_type)
    for key in analytical_data.keys():
        try:
            analytical_data[key] = analytical_data[key].loc[stone_id:stone_id]
        except (KeyError, TypeError):
            analytical_data[key] = pd.DataFrame()
            continue
        print("getting rid of excess columns in XRF")
        if key=="ed":
            ed_ind = ['SnO2', 'MnO', 'Fe2O3', 'WO3', 'Ga2O3', 'MoO3', 'PbO', 'MgO', 'Al2O3',
       'SiO2', 'TiO2', 'V2O3', 'Cr2O3', 'NiO', 'Ir', 'Pt', 'ZrO2', 'Total',
       'Nb2O5', 'Na2O', 'K2O', 'CaO', 'Cs2O', 'ZnO', 'Rb2O', 'Cl']
            ed_ind = [("Concentration", x) for x in ed_ind]
            analytical_data[key] = analytical_data[key].loc[:,ed_ind]
                
    return analytical_data


def dummy_load_model():
    path = "outputs/2022-05-13/17-43-13/model"
    cfg = gemtelligence.learning.utils.return_cfg_from_model_path(path)
    model = hol_net_v1.training.load_model(cfg, path)
    model = model.eval()
    return cfg, model

def single_pass_inference(report_nr, stone_data, stone_type, target):
    if stone_type=="sapphire":
        cfg, model = dummy_load_model()
        #model, cfg = load_model(stone_type, target, model_type)
        dataset_class = deep_learning_libs.data.CombinedDataset(stone_data, cfg, "test")
        #extract ID from the dictionary and assert that all IDs are the same
        inputs = dataset_class.get_test_sample(report_nr)
        
        for source in inputs.keys():
            tmp = torch.tensor(inputs[source]).float()
            assert tmp.isnan().sum() == 0 and tmp.isinf().sum() == 0
            inputs[source] = tmp
        logits = model(inputs)
        probs = softmax(logits, dim=1).detach().numpy()
        probs_mean = probs.mean(axis=0)
        target_list = utils.target_list(target.lower(), stone_type)
        return dict(zip(target_list, probs_mean))
        
    
def inference(stone_id, stone_type, target, is_recheck, ):
    stone_data = get_data(stone_id, stone_type, is_recheck)
    single_pass_inference(stone_data, stone_type, target)


if __name__ == "__main__":
    inference(19120122, "sapphire", "origin", [])
    