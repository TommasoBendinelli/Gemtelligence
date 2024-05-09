import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch
from gemtelligence import utils
from gemtelligence import learning
import pytorch_lightning as pl
import random
from collections import Counter
import glob
import hydra
from PIL import Image
from collections import defaultdict
import os
from torchvision import transforms
from pathlib import Path


def final_validation_processing(entries, sources):
    df_tot = pd.DataFrame()
    df_tot.index.name = "ID"
        
    # Set the name of the index to ID for all the dataframes
    for k,v in entries.items():
        v.index.name = "ID"

    for key in entries.keys():
        df_tot = df_tot.join(entries[key],how="outer",on="ID")
    df_tot.columns = pd.MultiIndex.from_tuples([(k,column) for k, df in entries.items() for column in df])
    # Creating the cartesian product 
    # df_tot = pd.concat(list(entries.values()), axis=1, join="outer", keys=entries.keys())
    # Retrieve the original values        
    # Keep samples below 50, otherwise might be cuda errors    
    if len(df_tot) > 50:      
        df_tot = df_tot.sample(50)
    
    fin_entries = {}
    
    if "uv" in sources:
        uv_values = np.array(df_tot[["uv"]]).astype(np.float)
        tmp = uv_values
        uv_values = np.reshape(uv_values, (uv_values.shape[0], 2, uv_values.shape[1]//2))
        fin_entries["uv"] = uv_values
        assert all(tmp[0, :uv_values.shape[2]] == uv_values[0][0])
        assert all(tmp[0, uv_values.shape[2]:] == uv_values[0][1])

    if "ed" in sources:
        ed_values = np.array(df_tot[["ed"]]).astype(np.float)
        fin_entries["ed"] = ed_values
        assert len(entries["ed"].columns) == ed_values.shape[-1]

    if "ftir" in sources:
        ftir_df = np.array(df_tot[["ftir"]]).astype(np.float)[:, None,:]
        fin_entries["ftir"] = ftir_df

    
    if "icp" in sources:

        icp_values = np.array(df_tot[["icp"]]).astype(np.float)
        fin_entries["icp"] = icp_values
        assert len(entries["icp"].columns) == icp_values.shape[-1]
            
    # Assert that all entries in fin_entries are of the same length
    assert len(set([len(x) for x in fin_entries.values()])) == 1
    
    return fin_entries



def get_uv(ID, uv_df, is_training, prob_to_ignore=0):
    """
    Get the uv values for the given ID ready for training or validation
    """
    if is_training:
        if ID in uv_df.index and random.random() > prob_to_ignore:
            curr = uv_df.loc[ID].fillna(0)
            if type(curr) == pd.Series: # It means that there is just an unique row
                uv_values = curr
            elif type(curr) == pd.DataFrame:
                uv_values = curr.sample().iloc[0]
            else:
                raise Exception("Bug in data.py")
            
            uv_values = np.array(uv_values.loc[["A","B"]]).astype(np.float)
        else:
            uv_values = np.array([0.0] *  len(uv_df[["A","B"]].columns))#pd.DataFrame({x:[0] for x in uv_df.columns}, index=[ID])

        uv_values = np.reshape(uv_values, (2, uv_values.shape[0]//2))
        return uv_values

    else:
        if ID in uv_df.index and random.random() > prob_to_ignore:
            uv_df = uv_df.loc[ID:ID][["A","B"]].fillna(0.0)
            uv_df.columns = ["uv_" + str(idx) for idx, x in enumerate(uv_df.columns)]
        else:
            uv_df = pd.DataFrame({f"uv_{x}":[0] for x in range(len(uv_df[["A","B"]].columns))}, index=[ID]) #np.array([0] *  len(uv_df[["A","B"]].columns))#pd.DataFrame({x:[0] for x in uv_df.columns}, index=[ID])
        return uv_df

def get_ed(ID, ed_df, is_training, prob_to_ignore=0, is_log=False, is_z_normalized=False, mean=None, std=None):
    """
    Get the uv values for the given ID ready for training or validation
    """
    if is_training:
        if ID in ed_df.index and random.random() > prob_to_ignore:
            curr = ed_df.loc[ID] #.fillna(0)
            if type(curr) == pd.Series: # It means that there is just an unique row
                ed_sample = curr
            elif type(curr) == pd.DataFrame:
                ed_sample = curr.sample().iloc[0]
            else:
                raise Exception("Bug in data.py")
            ed_values = np.array(ed_sample.loc["Concentration"]).astype(np.float)
        else:
            ed_values =  np.array([0.0] *  len(ed_df["Concentration"].columns)) 
        tmp = ed_values
    else:
        if ID in ed_df.index and random.random() > prob_to_ignore:
            ed_df = ed_df.loc[ID:ID]["Concentration"]
            ed_df.columns = ["ed_" + str(idx) for idx, x in enumerate(ed_df.columns)]

        else:
            ed_df =  pd.DataFrame({"ed_" + str(x): 0 for x in range(len(ed_df["Concentration"].columns))}, index=[ID])
        tmp = ed_df
    
    if is_z_normalized:
        tmp = (tmp - mean.values) / std.values
        # Replace nan with 0
        tmp[np.isnan(tmp)] = 0

    if is_log:
        tmp = np.abs(tmp) +1
        tmp = np.log(tmp)
    return tmp


def get_icp(ID, icp_df, is_training, prob_to_ignore=0, is_log=False):
    """
    Get the icp values for the given ID ready for training or validation
    """
    if is_training:
        if ID in icp_df.index and random.random() > prob_to_ignore:
            curr = icp_df.loc[ID]
            #curr = icp_df.loc[ID].apply(pd.to_numeric,errors="coerce").fillna(0)
            if type(curr) == pd.Series: # It means that there is just an unique row
                icp_values = curr
            elif type(curr) == pd.DataFrame:
                icp_values = curr.sample().iloc[0]
            else:
                raise Exception("Bug in data.py")
            icp_values = np.array(icp_values).astype(np.float)
        else:
            icp_values =  np.array([0.0] *  len(icp_df.columns)) 
        tmp = icp_values
    else:
        if ID in icp_df.index and random.random() > prob_to_ignore:
            icp_df = icp_df.loc[ID:ID].apply(pd.to_numeric,errors="coerce")
            icp_df.columns = ["icp_" + str(idx) for idx, x in enumerate(icp_df.columns)]

        else:
            icp_df =  pd.DataFrame({"icp_" + str(x): 0 for x in range(len(icp_df.columns))}, index=[ID])
        tmp = icp_df
        

    if is_log:
        icp_df = np.abs(tmp) +1
        icp_df = np.log(icp_df)
    else:
        icp_df = tmp
    return icp_df

def get_ftir(ID, ftir_df, is_training, prob_to_ignore=0):
    """
    Get the uv values for the given ID ready for training or validation
    """
    # if ID == 19040272:
    #     breakpoint()
    if is_training:
        
        if ID in ftir_df.index and random.random() > prob_to_ignore:
            curr = ftir_df.loc[ID].fillna(0)
            if type(curr) == pd.Series: # It means that there is just an unique row
                ftir_df = curr
            elif type(curr) == pd.DataFrame:
                ftir_df = curr.sample().iloc[0]
            else:
                raise Exception("Bug in data.py")
            
            ftir_df = np.array(ftir_df).astype(np.float)
        else:
            ftir_df = np.array([0.0] *  len(ftir_df.columns))#pd.DataFrame({x:[0] for x in uv_df.columns}, index=[ID])

        # Clip the values between 0 and 10 
        ftir_df = np.clip(ftir_df, 0, 10)
        return ftir_df
    else:
        if ID in ftir_df.index and random.random() > prob_to_ignore:
            ftir_df = ftir_df.loc[ID:ID].fillna(0.0)
            ftir_df.columns = [0.0 for x in ftir_df.columns]
        else:
            ftir_df = pd.DataFrame({f"ftir_{x}":[0.0] for x in range(len(ftir_df.columns))}, index=[ID])#np.array([0] *  len(uv_df[["A","B"]].columns))#pd.DataFrame({x:[0] for x in uv_df.columns}, index=[ID])
        return ftir_df

def get_image(ID, image_df, is_training):
    """
    Get paths for a given image
    """
    # if ID == 19040272:
    #     breakpoint()
    if is_training:
        if ID in image_df.index:
            curr = image_df.loc[ID].fillna(0)
            if type(curr) == pd.Series: # It means that there is just an unique row
                image_df = curr
            elif type(curr) == pd.DataFrame:
                image_df = curr.sample().iloc[0]
            else:
                raise Exception("Bug in data.py")
            
        else:
            image_df  =pd.DataFrame(columns=["path"], index=[ID])
        
        return image_df

        # # Clip the values between 0 and 10 
        # ftir_df = np.clip(ftir_df, 0, 10)
        # return ftir_df
    else:
        if ID in image_df.index:
            curr = image_df.loc[ID:ID] 
        else:
            curr = pd.DataFrame(columns=["path"], index=[ID])#np.array([0] *  len(uv_df[["A","B"]].columns))#pd.DataFrame({x:[0] for x in uv_df.columns}, index=[ID])
        return curr



class CombinedDataset(data.Dataset):
    def __init__(self, data, cfg, mode, pretraining=False, mean=None, std=None):
        assert pretraining == False # We don't do any pretraining, the argument is just passed for compatibility
        assert len(data["val"]) == len(set(data["val"].index))
        
        data_df = data["val"]
        self.mean = None
        self.std = None
        for source in cfg.sources.keys():
            if "is_z_normalized" in cfg.sources[source] and cfg.sources[source].is_z_normalized:
                if source == "ed":
                    self.mean =  data[source]["Concentration"].mean()
                    self.std = data[source]["Concentration"].std()
        
                    #data[source]["Concentration"] = (data[source]["Concentration"] - data[source]["Concentration"].mean()) / data[source]["Concentration"].std()
        self.index_to_ID = [ID for ID in data_df.index]
        self.cfg = cfg
        self.mapper = utils.mapper(cfg.target, stone_type=cfg.stone_type, str2num=True)
        # if self.cfg.add_secondary_target:
        #     if self.cfg.target == "origin":
        #         self.mapper_sec = utils.mapper("treatment", stone_type=cfg.stone_type, str2num=True)
        #     else:
        #         self.mapper_sec = utils.mapper("origin", stone_type=cfg.stone_type, str2num=True)
        self.mode = mode
        self.data = data
        self.sources = set(cfg.sources.keys()) # The data source used for the current experiment

    def __getitem__(self, index):
        ID = self.index_to_ID[index]

        if self.cfg.target in ("origin", "pre_origin"):
            # if self.data["val"].loc[ID, "FinalOrigin"] == "not requested":
            #     y = -1
            # if ID in [15020165,19097156]:
            #     breakpoint()
            # # else:
            try:
                y = self.mapper(self.data["val"].loc[ID, "FinalOrigin"])
            except Exception as e:
                print("Unknown origin for ID:", ID)
                y = -1
            if self.mode == "train":
                if type(y) != int:
                    print("Errod with ID:", ID)
                
        elif self.cfg.target == "treatment":
            try:
                y = self.mapper(self.data["val"].loc[ID, "Heat Treatment Value"])
            except Exception as e:
                print("Unknown treatment for ID:", ID)
                y = -1
        elif self.cfg.target == "enhancement":
            y = self.mapper(self.data["val"].loc[ID, "Clarity Enhancement Value"])
        else:
            raise NotImplementedError("Target {} not implemented".format(self.cfg.target))

        y_sec = -1
        # if self.cfg.add_secondary_target:
        #     if self.cfg.target == "origin":
        #         if self.cfg.stone_type in ["ruby","sapphire"]:
        #             t = self.data["val"].loc[ID, "Heat Treatment Value"]
        #             if t != -1:
        #                 y_sec = self.mapper_sec(t)
        #         elif self.cfg.stone_type == "emerald":
        #             t = self.data["val"].loc[ID, "Clarity Enhancement Value"]
        #             if t != -1:
        #                 y_sec = self.mapper_sec(t)
        #         else:
        #             raise NotImplementedError("Target {} not implemented".format(self.cfg.target))
        #     elif self.cfg.target != "origin":
        #         t = self.data["val"].loc[ID, "FinalOrigin"]
        #         if t != -1:
        #             y_sec = self.mapper_sec(t)
        #     else:
        #         raise NotImplementedError("Target {} not implemented".format(self.cfg.target))
                
                
            
        if self.mode == "train":
            entries = self.get_training_sample(ID)
        elif self.mode == "val" or self.mode == "test":
            entries = self.get_validation_sample(ID)
        
        
        return entries, torch.tensor(y).long(), torch.tensor(y_sec), ID

    def get_test_sample(self, ID):
        return self.get_validation_sample(ID)

    def get_training_sample(self, ID):
        """
        Sample randomly for each data source
        """
        
        assert any([ID in self.data[x].index for x in self.sources]), "ID {} not found in any of the sources. Bug Somewhere!".format(ID)
        entries = {}
        if "uv" in self.sources:
            uv = get_uv(ID, self.data["uv"], is_training=True, prob_to_ignore=self.cfg.method.prob_to_drop)
            entries["uv"] = uv
        if "ed" in self.sources:
            ed = get_ed(ID, self.data["ed"], is_training=True,  prob_to_ignore=self.cfg.method.prob_to_drop, is_log=self.cfg.sources.ed.is_log, is_z_normalized=self.cfg.sources.ed.is_z_normalized, mean=self.mean, std=self.std)
            entries["ed"] = ed
        
        if "ftir" in self.sources:
            ftir = get_ftir(ID, self.data["ftir"], is_training=True,  prob_to_ignore=self.cfg.method.prob_to_drop)
            entries["ftir"] = ftir[None,:]
            assert not np.isinf(ftir[None,:]).any().any()
        
        if "icp" in self.sources:
            icp = get_icp(ID, self.data["icp"], is_training=True, prob_to_ignore=self.cfg.method.prob_to_drop, is_log=self.cfg.sources.icp.is_log)
            entries["icp"] = icp

        # Check if any entry has nans
        if any([np.isnan(x).any() for x in entries.values()]):
            breakpoint()
        return entries

    def get_validation_sample(self, ID):
        """
        Create the cartesian product of the possible values for each data source
        """
        assert any([ID in self.data[x].index for x in self.sources]), "ID {} not found in any of the sources. Bug Somewhere!".format(ID)
        entries = {}
        
        if "uv" in self.sources:
            if self.cfg.only_val.uv:
                prob_to_ignore = 0
            else:
                prob_to_ignore = 1
            entries["uv"] = get_uv(ID, self.data["uv"], prob_to_ignore=prob_to_ignore, is_training=False).drop_duplicates().iloc[:10]
        if "ed" in self.sources:
            if self.cfg.only_val.ed:
                prob_to_ignore = 0
            else:
                prob_to_ignore = 1
            entries["ed"] = get_ed(ID, self.data["ed"], is_training=False,  prob_to_ignore=prob_to_ignore, is_log=self.cfg.sources.ed.is_log, is_z_normalized=self.cfg.sources.ed.is_z_normalized, mean=self.mean, std=self.std).drop_duplicates().iloc[:10]
        if "ftir" in self.sources:
            if self.cfg.only_val.ftir:
                prob_to_ignore = 0
            else:
                prob_to_ignore = 1
                
            entries["ftir"] = get_ftir(ID, self.data["ftir"], is_training=False,prob_to_ignore=prob_to_ignore).drop_duplicates().iloc[:20] # We only use the first 50 ftir values to not go out of memory
        if "icp" in self.sources:
            if self.cfg.only_val.icp:
                prob_to_ignore = 0
            else:
                prob_to_ignore = 1
            
            entries["icp"] = get_icp(ID, self.data["icp"], is_training=False, prob_to_ignore=prob_to_ignore, is_log=self.cfg.sources.icp.is_log).drop_duplicates().iloc[:10]
        
        # Set the name of the index to ID for all the dataframes
        fin_entries = final_validation_processing(entries, self.sources)
        
        # for k,v in entries.items():
        #     v.index.name = "ID"


        # for key in entries.keys():
        #     df_tot = df_tot.join(entries[key],how="outer",on="ID")
        # df_tot.columns = pd.MultiIndex.from_tuples([(k,column) for k, df in entries.items() for column in df])
        # # Creating the cartesian product 
        # # df_tot = pd.concat(list(entries.values()), axis=1, join="outer", keys=entries.keys())
        # # Retrieve the original values        
        # # Keep samples below 50, otherwise might be cuda errors    
        # if len(df_tot) > 50:      
        #     df_tot = df_tot.sample(50)
      
        # fin_entries = {}
        # if "uv" in self.sources:
        #     uv_values = np.array(df_tot[["uv"]]).astype(np.float)
        #     tmp = uv_values
        #     uv_values = np.reshape(uv_values, (uv_values.shape[0], 2, uv_values.shape[1]//2))
        #     fin_entries["uv"] = uv_values
        #     assert all(tmp[0, :uv_values.shape[2]] == uv_values[0][0])
        #     assert all(tmp[0, uv_values.shape[2]:] == uv_values[0][1])

        # if "ed" in self.sources:
        #     ed_values = np.array(df_tot[["ed"]]).astype(np.float)
        #     fin_entries["ed"] = ed_values
        #     assert len(entries["ed"].columns) == ed_values.shape[-1]

        # if "ftir" in self.sources:
        #     ftir_df = np.array(df_tot[["ftir"]]).astype(np.float)[:, None,:]
        #     fin_entries["ftir"] = ftir_df

        
        # if "icp" in self.sources:
        #     icp_values = np.array(df_tot[["icp"]]).astype(np.float)
        #     fin_entries["icp"] = icp_values
        #     assert len(entries["icp"].columns) == icp_values.shape[-1]
                
        # # Assert that all entries in fin_entries are of the same length
        # assert len(set([len(x) for x in fin_entries.values()])) == 1
        
        return fin_entries

    def __len__(self):
        return len(self.index_to_ID)



class DataModule(pl.LightningDataModule):
    def __init__(
        self, data, dataset_class, cfg, number_of_workers=0, shuffle_train=True, ispretraining=False, ID_train=None, ID_val = None, ID_test=None,
    ):
        super().__init__()
        self.data = data
        # Assert that val dataframe contains unique IDs
        assert len(data["val"]) == len(set(data["val"].index))
        #curr = data[~data.index.duplicated(keep='first')]
        val_df = data["val"]

        index = [idx for idx, _ in enumerate(val_df.index)]
        country_val_counter = Counter(val_df["FinalOrigin"].values).most_common()
        min_country_val = country_val_counter[-1]
        min_num_samples = min_country_val[1]

        self.train_data = learning.utils.loc_stones(data, ID_train)


        self.val_data = learning.utils.loc_stones(data, ID_val)
        if ID_test:
            self.test_data = learning.utils.loc_stones(data, ID_test)
        else:
            self.test_data = None            
   
        assert not learning.utils.obtain_all_stones_available_in_sources(self.train_data) & learning.utils.obtain_all_stones_available_in_sources(self.val_data) 
        #assert not set(self.train_data.index) & set(self.val_data.index)
        self.number_of_workers = cfg.run.num_workers
        self.cfg = cfg
        self.shuffle_train = shuffle_train
        self.dataset_class = dataset_class
        self.pretraining = ispretraining
        self.batch = cfg.method.batch


    def setup(self, stage=None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            self.shuffle=True
            self.drop_last=True
            self.training_dataset = self.dataset_class(
                data=self.train_data,
                cfg=self.cfg,
                mode="train",
                pretraining=self.pretraining
            )
            self.val_dataset = self.dataset_class(
                data=self.val_data,
                cfg=self.cfg,
                mode="val",
                pretraining=self.pretraining
            )
            
            if self.test_data:
                self.test_dataset = self.dataset_class(
                    data=self.test_data,
                    cfg=self.cfg,
                    mode="val",
                    pretraining=self.pretraining
                )
                    
            
            
        else:
            self.shuffle=False
            self.drop_last=False
            self.training_dataset = self.dataset_class(
                data=self.train_data,
                cfg=self.cfg,
                mode="train",
                pretraining=self.pretraining
            )
            self.val_dataset = self.dataset_class(
                data=self.val_data,
                cfg=self.cfg,
                mode="val",
                pretraining=self.pretraining
            )
            if self.test_data:
                self.test_dataset = self.dataset_class(
                    data=self.test_data,
                    cfg=self.cfg,
                    mode="val",
                    pretraining=self.pretraining
                )
            
            

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.number_of_workers,
            worker_init_fn=lambda x: np.random.seed(0 + x),
            collate_fn=self.custom_collate_train,
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        valloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.number_of_workers,
            worker_init_fn=lambda x: np.random.seed(0 + x),
            collate_fn=self.custom_collate_val,

        )
        return valloader

    def test_dataloader(self):
        """returns validation dataloader"""
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.number_of_workers,
            worker_init_fn=lambda x: np.random.seed(0 + x),
            collate_fn=self.custom_collate_val,
        )
        return testloader

    def custom_collate_val(self, x):
        """
        Just use for validation and testing. It contains all the samples from that ID.
        """
        len_batch = 1
        assert len(x) == len_batch# Batch consists of only one sample in Validation and Test
        inputs = x[0][0]
        y = x[0][1]
        y_sec = x[0][2]
        ID = x[0][3]
        samples_set = set([source.shape[0] for source in inputs.values()])
        assert len(samples_set) ==1 
        number_of_samples = samples_set.pop()

        # n_sources = len(input)
        for source in inputs.keys():
            tmp = torch.tensor(inputs[source]).float()
            assert tmp.isnan().sum() == 0 and tmp.isinf().sum() == 0
            inputs[source] = tmp
            
        ys = torch.stack([torch.tensor(int(y)) for _ in range(number_of_samples)])
        y_sec = torch.stack([torch.tensor(int(y_sec)) for _ in range(number_of_samples)])
        IDs = [ID for _ in range(number_of_samples)]
        
        return inputs, ys, y_sec, IDs

    def custom_collate_train(self, x):
        """
        Just use for training. It contains all the samples from that ID.
        """
        number_of_samples = len(x)

        # breakpoint()
        inputs_raw = [i[0] for i in x]
        inputs = {}
        ys = [i[1] for i in x]
        y_secs = [i[2] for i in x]
        IDs = [i[3] for i in x]
        # samples_set = set([source.shape[0] for source in inputs.values()])
        # assert len(samples_set) ==1 
        

        # # n_sources = len(input)
        assert len(set([tuple(x.keys()) for x in inputs_raw])) ==1
        sources = inputs_raw[0].keys()
       
        for source in sources:
            curr = []
            for x in inputs_raw:
                curr.append(x[source])
            tmp = torch.tensor(np.array(curr)).float()
            assert tmp.isnan().sum() == 0 and tmp.isinf().sum() == 0
            inputs[source] = tmp
        # for source in inputs.keys():
        #     inputs[source] = torch.tensor(inputs[source]).float()
        ys = torch.stack([torch.tensor(int(y)) for y in ys])
        y_secs = torch.stack([torch.tensor(int(y_sec)) for y_sec in y_secs])
        IDs = [ID for ID in IDs]
        return inputs, ys, y_secs, IDs