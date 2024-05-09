import pickle
from typing import List
from dataclasses import dataclass
import logging
from gemtelligence import utils
from gemtelligence import learning
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datetime import date
from omegaconf import DictConfig
import hydra

def runner(cfg, client):
    """
    Define the cross validation scheme and run the training. The result is a pickle file containing the results for each cv fold.
    """
    #Preparing CV scheme
    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)        
    res = learning.utils.cv_training(cv_scheme, client, cfg, split_val_test=cfg.split_val_test)
    with open(
        "results".format(),
        "wb",
    ) as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

def init_command_line_args(cfg):
    """
    Check validity of command line arguments and change stuff inplace
    """
    available_sources = {"ed","icp", "ftir", "uv"}
    #Prepare data of the correct type:
    # Ugly hack, if there square brackets [ ] in source name, then it is a list of sources so we need to convert it to a list via eval.
    # Otherwise, we treat it as a string
    assert len(cfg.sources), "At least one data sources is required"
    for key in cfg.sources.keys():
        assert key in available_sources

    if "ed" in cfg.sources:
        cfg.sources.ed.num_columns=26

            
    if "icp" in cfg.sources:
        cfg.sources.icp.num_columns=16
           

@hydra.main(config_name="config", config_path="runner")
def main(cfg: DictConfig) -> None:
    """
    Entry point for all training algorithms for any stone, but ensemble algorithm.
    Initilize the command line arguments, load the data, and run the training algorithm.
    """
    init_command_line_args(cfg)
    #Loading data
    # If it is sapphire, we load all the data (since the df is not too big)
    client_data = utils.load_client_data(stone_type = cfg.stone_type)
    
    #Create Data specific Data Structure
    data = Data(dict_df_client=client_data, dict_df_reference=None)

    runner(cfg, data)
    
    return 0





@dataclass
class Data:
    dict_df_client: dict
    dict_df_reference: dict
    main_features: List = None
    all_features: List = None



if "__main__" == __name__:
    result = main()
