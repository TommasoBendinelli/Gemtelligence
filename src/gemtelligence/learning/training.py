import torch
import numpy as np
from torch.nn.functional import softmax
from gemtelligence import learning
from gemtelligence import utils
import pytorch_lightning as pl
import pandas as pd
from collections import defaultdict
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import hydra

def are_two_stones_a_subset_of_each_other(df_dict, stones,second_key):
    to_remove = set()
 
    for stone in stones:
        # if stone == candidate_test:
        #     continue
        to_check_against = stones - {stone} - to_remove
        
        for entry in to_check_against:
            if second_key == "ed":
                keys = ["ed","icp","uv"]
            elif second_key == "ftir":
                keys = ["ftir","uv"]
                
            is_subset = True
            for key in keys:
                if (stone in df_dict[key].index) and (entry in df_dict[key].index):
                    
                    stone_entry = df_dict[key].loc[stone:stone]
                    
                    # Sort them by the first and second column
                    stone_entry = stone_entry.sort_values(by=[stone_entry.columns[0],stone_entry.columns[1]])
                    
                    
                    to_check_against_entry = df_dict[key].loc[entry:entry]
                    to_check_against_entry = to_check_against_entry.sort_values(by=[to_check_against_entry.columns[0],to_check_against_entry.columns[1]])

                    # Compare if stone_entry.values are equivalent to to_check_against_entry.values
                    if len(stone_entry.values) != len(to_check_against_entry.values):
                        is_subset = False
                        break 
                    if not (stone_entry.values == to_check_against_entry.values).all():
                        is_subset = False
                        break
                
                elif (stone in df_dict[key].index) and (entry not in df_dict[key].index):
                    is_subset = False
                    break
                
            if is_subset:
                to_remove.add(stone)
                break
            
        if is_subset:
            to_remove.add(stone)
                    
    # if 13080078 in stones:
    #     breakpoint()                
    return to_remove                    


    


@utils.create_subfolder
def runner(data, cfg, counter, batch=64, epochs=100, num_workers=6) -> Tuple[pd.Series]:
    """
    Neural network based on PyTorch
    """
    if cfg.save_best_model_in_val:
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            filename='{epoch}-{val_acc:.2f}',
            monitor=cfg.to_monitor,
            mode=cfg.to_monitor_mode,
    )
    else:
        model_checkpoint = None
            
    
    model = learning.hol_net_v1.load_model(cfg)
    client_data = data.dict_df_client
    
    tmp = utils.load_client_data("sapphire")
    
    for key in ["uv","ed","icp","ftir"]:
        # Remove rows that are all nan
        is_all_nan = tmp[key].isna().all(axis=1)
        tmp[key] = tmp[key][~is_all_nan]
    #     breakpoint()
    # tmp["ed"] = tmp["ed"].dropna()
    # tmp["uv"] = tmp["uv"].dropna()
    
    possible_candidates = set()
    # Find all recheck stones in the test set
    for key, entry in tmp["val"].iterrows():
        if key in data.test_ids:
            continue
        
        if data.test_ids & set(entry["Rechecks"]):
            all_rechecks =  set(entry["Rechecks"]) #- set(data.test_ids) #possible_candidates |  # Possible IDs of rechecks that are not in the test set
            #candidate_test = list(data.test_ids & set(entry["Rechecks"]))[0] # Stone in the test set that has rechecks

            #to_iterate = currs
            if cfg.target in ["origin","pre_origin"]:
                second_key = "ed"
            elif cfg.target == "treatment":
                second_key = "ftir"
                
            # if not (candidate_test in tmp["uv"].index or candidate_test in tmp[second_key].index):
            #     continue
            stones_removed = are_two_stones_a_subset_of_each_other(tmp, all_rechecks,second_key)
            currs = all_rechecks - set(stones_removed)
            # if 13080078 in currs:
            #     breakpoint()
            if cfg.is_consistent_exp == True:
                for curr in currs:
                    # Add to the new stones the old data 
                    stones_subset_to_be_considered = [x for x in currs if x < curr] 
                    for idx, xx in enumerate(stones_subset_to_be_considered):
                        for key in client_data.keys():
                            if key == "val":
                                continue
                            
                            
                            
                            assert curr in all_rechecks and xx in all_rechecks
                            if not curr in tmp[key].index:
                                if not xx in tmp[key].index:
                                    continue
                                
                                new_data_to_add = tmp[key].loc[xx:xx]
                                # Change the ID of the stone to the new one
                                new_data_to_add.index = [curr for _ in range(len(new_data_to_add))]
                                # if key == "icp":
                                #     breakpoint()
                                # Concatenate the new stone with the old one
                                tmp[key] = pd.concat([tmp[key], new_data_to_add],axis=0)
                                #client_data[key] = pd.concat([client_data[key], tmp[key].loc[xx]],axis=0)
                                
                            #client_data[key] = pd.concat([tmp[key].loc[stone]],axis=0)

            possible_candidates.update(currs)
            #possible_candidates.add(candidate_test)
    
    #breakpoint()
    if cfg.is_consistent_exp == True:
        data.test_ids = possible_candidates
        
        # TODO: Move this in data validation
        tmp_icp = tmp["icp"].apply(pd.to_numeric,errors="coerce")
        
        for col in tmp["icp"].columns:
            tmp_icp = tmp_icp.loc[~(tmp_icp[col] <0)]

        tmp_icp = tmp_icp.loc[~(tmp_icp["Al27"] < 500000)]
        tmp_icp = tmp_icp.loc[~(tmp_icp["Nd146"] > 10)]
        tmp["icp"] = tmp_icp
    
        tmp_post_processed = learning.utils.final_postprocessing(tmp, cfg)
        client_data = tmp_post_processed

    
    dataset_class = learning.data.CombinedDataset
    # Used only for finding the best model
    if cfg.target == "origin":
        filtered_df = utils.return_df_for_paper(client_data.copy(), "origin", filter_noisy=True)
        filtered_val_ids = set(data.val_ids) & set(filtered_df["val"].index)
    elif cfg.target == "treatment":
        filtered_df = utils.return_df_for_paper(client_data.copy(), "treatment", filter_noisy=True)
        filtered_val_ids = set(data.val_ids) & set(filtered_df["val"].index)
    elif cfg.target == "pre_origin":
        #filtered_df = utils.return_df_for_paper(client_data.copy(), "pre_origin", filter_noisy=True)
        filtered_val_ids = set(data.val_ids) #& set(filtered_df["val"].index)    
    
    else:
        raise NotImplementedError
    
    datamodule_instance = learning.data.DataModule(client_data, dataset_class, cfg, number_of_workers=cfg.run.num_workers, 
                                            shuffle_train=True, ID_train=data.train_ids,ID_val=filtered_val_ids, ID_test=data.test_ids)
    
    check_val_every_n_epoch = cfg.method.check_val_every_n_epoch

    trainer = pl.Trainer(
                gpus=cfg.run.gpu, 
                max_epochs=cfg.method.max_epoch, 
                callbacks=[model_checkpoint], 
                precision=cfg.run.precision, 
                gradient_clip_val=1, 
                check_val_every_n_epoch=check_val_every_n_epoch)
            
    if (not "only_val" in cfg or not cfg.only_val.path):
        trainer.fit(model,datamodule_instance)
        model_path = model_checkpoint.best_model_path
    elif cfg.only_val.path:
        # List the model in the folder
        folder_path =  Path(hydra.utils.to_absolute_path(cfg.only_val.path)) / f"{counter}" / f"lightning_logs/version_0/checkpoints"
        model_path = list(folder_path.glob("*.ckpt"))[0]

    datamodule_instance = learning.data.DataModule(client_data, dataset_class, cfg, number_of_workers=cfg.run.num_workers, 
                                            shuffle_train=True, ID_train=data.train_ids,ID_val=data.val_ids, ID_test=data.test_ids)
    datamodule_instance.setup(stage="test")
    print("Computing Confusion Matrix")

    if cfg.save_best_model_in_val:
        print("Loading best model")
        model = learning.hol_net_v1.load_model(cfg, model_path)
        model = model.eval()
    else:
        model = model.eval()
    model = model.cuda()
    counter = 0
    
    if cfg.is_consistent_exp != True:
        prob_train, _ = multiple_instances_handler(datamodule_instance.train_dataloader,model,cfg)
        prob_val, counter = multiple_instances_handler(datamodule_instance.val_dataloader,model,cfg)        
        print(f"Computed score val {counter/len(prob_val)}")
    else:
        prob_train = None
        prob_val = None
        
    prob_test, counter = multiple_instances_handler(datamodule_instance.test_dataloader,model,cfg)

    
    print(f"Computed score test {counter/len(prob_test)}")

    return prob_train, prob_val, prob_test


def validate(data, model, cfg):
    """
    Neural network based on PyTorch
    """
    
    client_data = data.dict_df_client
    
    dataset_class = learning.data.CombinedDataset
    
    datamodule_instance = learning.data.DataModule(client_data, dataset_class, cfg, number_of_workers=cfg.run.num_workers, 
                                            shuffle_train=True, ID_train=data.train_ids,ID_val=data.val_ids,ID_test=data.test_ids)
   
        
    datamodule_instance.setup(stage="test")
    print("Computing Confusion Matrix")
    # model = model
    model = model.eval()
    model = model.cuda()
    counter = 0

    prob_train, _ = multiple_instances_handler(datamodule_instance.train_dataloader,model,cfg)
    prob_test, counter = multiple_instances_handler(datamodule_instance.val_dataloader,model,cfg)


    print(f"Computed score {counter/len(prob_test)}")

    return prob_train, prob_test


def multiple_instances_handler(iterator, model,cfg):
    model.zero_grad()
    res = defaultdict(list)
    y_gt = {} 
    fin = {}
    counter = 0 
    torch.cuda.empty_cache()
    print(torch.cuda.max_memory_allocated())
    for x_raw, y_raw, _, ID_raw in tqdm(iterator()):
        # Split the batch into 1 elements chunks for avoiding out of memory error
        for i in range(0, len(ID_raw), 1):
            y = y_raw[i:i+1]
            ID = ID_raw[i:i+1]

            # if cfg.is_reference_include:
            #     to_test = str(ID[0])
            #     if not str(ID[0]).isdigit() or len(to_test) != 8:
            #         continue # Reference stone -> We don't want to test it
            x = {key: source[i:i+1].cuda()for key, source in x_raw.items()}
            x = {key: source.detach() for key, source in x.items()}

            if cfg.method.name == "saint":
                pred = model(x)[0]
            else:
                pred = model(x)
           
            select = pred[:,:len(list(utils.target_list(cfg.target,cfg.stone_type)))]
            curr = softmax(select,dim=1 ).cpu().detach().numpy()
            del x 
            torch.cuda.empty_cache()
            for idx, _ in enumerate(curr):
                res[int(ID[0])].append(curr[idx])
                y_gt[int(ID[0])] = int(y[0])
    for ID in res.keys():
        curr = np.array(res[ID])
        if cfg.method.val_method == "max":
            comp = curr.max(axis=1)
            idx = np.argmax(comp)
            curr = curr[idx,:]
        elif cfg.method.val_method == "mean":
            curr = np.array(res[ID])
            comp = curr.mean(axis=0)
            curr = comp
        else: 
            raise KeyError
        counter = sum(np.array(y_gt[ID] == curr.argmax())[None]) + counter
        fin[ID] = curr

    df = pd.DataFrame.from_dict(fin,orient="index")
    df_y_gt = pd.DataFrame.from_dict(y_gt,orient="index")
    df_fin = pd.concat([df,df_y_gt],axis=1)
    return df_fin, counter


