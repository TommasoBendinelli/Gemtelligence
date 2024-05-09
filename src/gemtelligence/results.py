from collections import defaultdict
from typing import List, Tuple
from . import utils 
from numpy.core.fromnumeric import argmax
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, log_loss
import numpy as np
import pandas as pd
from copy import deepcopy

class Results:
    """
    This class holds the results of each experiment (5 CV folds). It can then be visualized via the visualization methods.
    """
    def __init__(self, data, cfg, df=None):
        self.data = deepcopy(data)
        if type(df) != type(None):
            self.data.dict_df = df
        val = [("Considered","Validation"),("Considered","Training"),("Considered","Reviewed"),("Considered","Recheck")]

        targets = [("Pred",x) for x in utils.target_list(cfg.target,cfg.stone_type)]
        c =  val + [
            ("ID", "ID")]+ targets \
            +[ 
            (cfg.target, "Predicted"),
            (cfg.target, "GT"),
            (cfg.target, "Matched"),
        ]

        self.col = pd.MultiIndex.from_tuples(c)
        #self.df = pd.DataFrame(columns=self.col)
        self.model = []
        self.cfg = cfg
        #self.label_enc = data.label_enc
        self.df_training = pd.DataFrame(columns=self.col)
        self.df_validation = pd.DataFrame(columns=self.col)
        self.df_test = pd.DataFrame(columns=self.col)
    
    def _confusion_matrix(self, pd_section, threshold=0, normalize=None, one_per_sample=False):
        # if self.cfg.target == "Heat Treatment Value":
        #     labels = self.label_enc.transform([0,1])
        if one_per_sample:
            pred, gt = [], []
            for id, group in pd_section.groupby([("ID","ID")]):
                prediction = group["Pred"].values
                row, _ = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape)
                pred.append(prediction[row])
                gt.append(group[[self.cfg.target, "GT"]].iloc[0])
            pred = np.array(pred)
            gt = np.array(gt)
        else:
            pred = pd_section["Pred"].values
            if "GT" in set(pd_section[self.cfg.target].columns):
                gt = pd_section[self.cfg.target]["GT"].values
            # Handling old code
            elif "Origin" in set(pd_section[self.cfg.target].columns):
                gt = pd_section[self.cfg.target]["Origin"].values
            elif "Heat" in set(pd_section[self.cfg.target].columns):
                gt = pd_section[self.cfg.target]["Heat"].values
            else:
                raise ValueError(f"No column that can be used found in {set(pd_section[self.cfg.target].columns)}")  
        bool_cond = np.max(pred, axis=1) < threshold
        _, indeces_below_threshold = np.where([bool_cond])
        
        gt = gt[np.logical_not(bool_cond)]
        pred = pred[np.logical_not(bool_cond)]
        # check if map is a method of the class
        if "map" in dir(self):
            gt= self.map(gt)
            labels = [x for x in range(len(utils.target_list(self.cfg.target,self.cfg.stone_type)))]
        else:
            mapper = utils.mapper(self.cfg.target.lower(), 'sapphire', str2num=True)
            gt= mapper(gt)
            labels = [x for x in range(len(utils.target_list(self.cfg.target.lower(),'sapphire')))]
        # gt = utils.country_mapper(gt, self.cfg.stone_type)
        #gt = self.label_enc.transform(gt)
        cf = confusion_matrix(
            gt, np.argmax(pred, axis=1), labels=[x for x in range(len(labels))], normalize=normalize
        )
        cf = np.round(cf, 2)
        return cf, indeces_below_threshold

    def confusion_matrices(self, threshold=0, normalize=None, one_per_sample=False, df=None, exclude_false_val=True) -> Tuple[List, List]:
        """Given the result dataframe return a list of confusion matrices, each element corresponds to a CV fold"""
        if type(df) == type(None):
            df = self.df.copy()
        if exclude_false_val:
            df = self.return_ok_validation(df)
        #if self.cfg.target == "origin":
            #df = east_africa_fixer(df, "origin")
        confusion_matrices = []
        indices_list = []
        cv = max(self.df.index.get_level_values(0))
        assert self.cv_counter - 1 == cv
        for i in range(self.cv_counter):
            cf, indices_below_threshold = self._confusion_matrix(
                df.loc[i], threshold=threshold, normalize=None, one_per_sample=one_per_sample
            )
            confusion_matrices.append(cf)
            indices_list.append(indices_below_threshold)
        if not one_per_sample:
            assert sum(sum(sum(np.array(confusion_matrices)))) == (
                len(df) - sum(len(x) for x in indices_list)
            )
        return confusion_matrices, indices_list

    def return_subsect_roll(self,threshold, upper_bound=None, mode=None):
        if not mode:
            df = self.df 
        elif mode == "val":
            df = self.df_validation
        elif mode == "train":
            df = self.df_training
        elif mode == "test":
            df = self.df_test
        else: 
            raise ValueError(f"Mode {mode} not recognized")
            
        
        if upper_bound:
            bool_cond_l = np.max(df["Pred"].values, axis=1) > threshold
            bool_cond_r =  np.max(df["Pred"].values, axis=1) < upper_bound
            bool_cond = bool_cond_l & bool_cond_r
        else:
            bool_cond = np.max(df["Pred"].values, axis=1) > threshold
        #_, indeces_below_threshold = np.where([bool_cond])
        return df.loc[bool_cond]
        
   

class ResultsCV(Results):
    def __init__(self, data, cfg, df=None):
        super(ResultsCV, self).__init__(data,cfg,df)
        self.cv_counter = 0
        self.cv_res = []
        self.map = utils.mapper(cfg.target,cfg.stone_type,str2num=True) #str to num
        self.inv_map = utils.mapper(cfg.target,cfg.stone_type,str2num=False) #num to str
        self.target_tuple = utils.target_tuple(cfg.target)
        self.cfg = cfg

    def fill_results(self, test_index, predictions, mode):
        val_df = self.data.dict_df_client["val"]
        
        df = val_df.loc[test_index,[]]
        df["Considered_Validation"] = val_df.loc[test_index,("Considered_Validation")].values
        df["Considered_Training"] = val_df.loc[test_index,("Considered_Training")].values
        df["Reviewed"] = val_df.loc[test_index,("Reviewed")].values
        df["Recheck"] = np.nan
        df["ID"] = df.index
        
        for t in utils.target_list(self.cfg.target, self.cfg.stone_type):
            #We need to look at the column of the prediction that has the index corresponding to the target 
            df[t] = predictions[:,self.map(t)]

        y = np.argmax(predictions, axis=1)
        predictions = self.inv_map(y)

        df["Predicted"] = predictions

        df[self.target_tuple] = val_df.loc[test_index, self.target_tuple]
        df["Matched"] = df["Predicted"] == df[self.target_tuple]
        df.columns = self.col
        r = [(self.cv_counter, x) for x in df.index]
        df.index = pd.MultiIndex.from_tuples(r, names=["CV", "Index"])
        if mode == "train":
            self.df_training = pd.concat([self.df_training, df])
            self.df_training.index = pd.MultiIndex.from_tuples(self.df_training.index, names=["CV", "Index"])
        elif mode == "val": 
            self.df_validation = pd.concat([self.df_validation, df])
            self.df_validation.index = pd.MultiIndex.from_tuples(self.df_validation.index, names=["CV", "Index"])
        elif mode == "test":
            self.df_test = pd.concat([self.df_test, df])
            self.df_test.index = pd.MultiIndex.from_tuples(self.df_test.index, names=["CV", "Index"])
        else:
            raise KeyError("Mode not recognized")
        
    def update_cv_res(self, cv_res):
        self.cv_res.append(cv_res)
        self.cv_counter += 1
        


    def return_only_stones_from(self,source="All",mode="train"):
        if mode == "train":
            df = self.df_training.copy()
        elif mode == "val":
            df = self.df_validation.copy()
        elif mode == "test":
            df = self.df_test.copy()
        else:
            raise KeyError("Mode not recognized")
        
        if source=="All":
            return self.df
        if source=="Intersection":
            return self.df.dropna()
        candidates = set(self.df.index)
        if "UV_DATA" in source:
            uv_col = [("FEATURES",('Prob', 'UV_DATA', x)) for x in ["Burma","Kashmir", "Madagascar", "Sri Lanka"]]
            new_candidates = self.df.loc[~self.df.loc[:,uv_col].isna().all(axis=1)].index
            candidates = candidates & set(new_candidates)
        if "ICP_DATA" in source:
            uv_col = [("FEATURES",('Prob', 'ICP_DATA', x)) for x in ["Burma","Kashmir", "Madagascar", "Sri Lanka"]]
            new_candidates = self.df.loc[~self.df.loc[:,uv_col].isna().all(axis=1)].index
            candidates = candidates & set(new_candidates)
        if "ED_DATA" in source:
            uv_col = [("FEATURES",('Prob', 'ED_DATA', x)) for x in ["Burma","Kashmir", "Madagascar", "Sri Lanka"]]
            new_candidates = self.df.loc[~self.df.loc[:,uv_col].isna().all(axis=1)].index
            candidates = candidates & set(new_candidates)
        return self.df.loc[candidates]

    def close_results(self):
        if len(self.df_training):
            self.df_training[(self.cfg.target, "Matched")] = self.df_training[(self.cfg.target, "Matched")].astype("bool")
        elif len(self.df_validation):
            self.df_validation[(self.cfg.target, "Matched")] = self.df_validation[(self.cfg.target, "Matched")].astype("bool")
        elif len(self.df_test):
            self.df_test[(self.cfg.target, "Matched")] = self.df_test[(self.cfg.target, "Matched")].astype("bool")
            
        del self.data

    def compute_F1_score(self, df=None, exclude_false_val = True):
        if type(df) == type(None):
            df = self.df.copy()
        if exclude_false_val:
            df = self.return_ok_validation(df)
        
        if "GT" in set(df[self.cfg.target].columns):
            gt = df[self.cfg.target]["GT"].values
            labels = utils.target_list(self.cfg.target.lower(),self.cfg.stone_type)
        elif "Origin" in set(df[self.cfg.target].columns):
            gt = df[self.cfg.target]["Origin"].values
            labels = utils.target_list(self.cfg.target.lower(),"sapphire")
        elif "Heat" in set(df[self.cfg.target].columns):
            gt = df[self.cfg.target]["Heat"].values
            labels = utils.target_list(self.cfg.target.lower(),"sapphire")
        print("Without east Africa")
        print(f1_score(
            gt,
            df[self.cfg.target, "Predicted"].values,
            average="macro",labels=labels,
        ))
        #if self.cfg.target == "origin":
            #df = east_africa_fixer(df, "origin")
        return f1_score(
            gt,
            df[self.cfg.target, "Predicted"].values,
            average="macro", labels=labels,
        )
    
    def compute_accuracy_score(self,df=None, exclude_false_val = True):
        if type(df) == type(None):
            df = self.df.copy()
        if exclude_false_val:
            df = self.return_ok_validation(df)
        #if self.cfg.target == "origin":
            #df = east_africa_fixer(df, "origin")
        
        # East Africa is a special case, if a stone from Madagascar, Monzabique, Tanzania, Malawi, Kenya or Sri Lanka is predicted as East Africa,
        # we consider it correcly predicted. Also the opposite is true.
        # For doing this we replace the origin of the stone if it is a country in the list of East Africa countries and if it is predicted as East Africa
        # and we replace it with the opposite country.
        
        if "GT" in set(df[self.cfg.target].columns):
            res = accuracy_score(np.array(df[self.cfg.target,"GT"]),np.array(df[self.cfg.target, "Predicted"]))  
        # Handling of deprecated code
        elif "Origin" in set(df[self.cfg.target].columns):
            res = accuracy_score(np.array(df[self.cfg.target,"Origin"]),np.array(df[self.cfg.target, "Predicted"]))  
        elif "Heat" in set(df[self.cfg.target].columns):
            res = accuracy_score(np.array(df[self.cfg.target,"Heat"]),np.array(df[self.cfg.target, "Predicted"]))  
        else:
            raise ValueError("No ground truth found in dataframe")
        return res 

    def compute_cross_entropy(self, df=None, temperature=1, exclude_false_val = True):
        if type(df) == type(None):
            df = self.df.copy()
        if exclude_false_val:
            df = self.return_ok_validation(df)
        #if self.cfg.target == "origin":
            #df = east_africa_fixer(df, "origin")
        prob = df["Pred"].values

        if "GT" in set(df[self.cfg.target].columns):
            gt = self.map(df[self.cfg.target]["GT"].values)
            labels=utils.target_list(self.cfg.target.lower(),self.cfg.stone_type)
        elif "Origin" in set(df[self.cfg.target].columns):
            mapper = utils.mapper(self.cfg.target.lower(), 'sapphire', str2num=True)
            gt = mapper(df[self.cfg.target]["Origin"].values)
            labels=utils.target_list(self.cfg.target.lower(),'sapphire')
        elif "Heat" in set(df[self.cfg.target].columns):
            mapper = utils.mapper(self.cfg.target.lower(), 'sapphire', str2num=True)
            gt = mapper(df[self.cfg.target]["Heat"].values)
            labels=utils.target_list(self.cfg.target.lower(),'sapphire')
        return log_loss(gt,prob,labels=labels)
    
    def return_ok_validation(self,df):
        return df.loc[df.loc[:,("Considered","Validation")] ==  True]

def return_corr_df(res:Results):
    df = res.df
    df.index = df.index.droplevel(0)
    return df


# def east_africa_fixer(df, target):
#     """
#     This function takes care of the prediction beloging to east africa and gt belonging to an east africa country or viceversa.
#     returns a dataframe where east africa is replaced with the correct country if it is predicted as east africa and viceversa.
#     """
#     # Case 1: GT in a East Africa country, Predicted in East Africa => We set GT as East Africa
#     bool_cond1 = df.loc[:,(target,"GT")].isin(["Madagascar", "Mozambique", "Tanzania", "Malawi", "Kenya", "Sri Lanka"])
#     bool_cond2 = df.loc[:,(target,"Predicted")] ==  "East Africa"
#     df.loc[bool_cond1 & bool_cond2, (target,"GT")] = df.loc[bool_cond1 & bool_cond2, (target,"Predicted")] 

#     # Case 2: GT is East Africa, Predicted as an East Africa country => We set GT as East Africa country
#     bool_cond1 = df.loc[:,(target,"GT")] ==  "East Africa"
#     bool_cond2 = df.loc[:,(target,"Predicted")].isin(["Madagascar", "Mozambique", "Tanzania", "Malawi", "Kenya", "Sri Lanka"])
#     df.loc[bool_cond1 & bool_cond2, (target,"GT")] = df.loc[bool_cond1 & bool_cond2, (target,"Predicted")] 
#     return df 
