from logging import error
from shutil import Error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from torch.nn.modules import loss
from gemtelligence import learning
import pandas as pd
from .. import utils
import numpy as np
from typing import Tuple
import hydra
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

def return_columns_source(X_train,cfg):
    """
    Selects the columns to be used for training from the dataframe.
    """
    columns = np.array([False for x in range(len(X_train.columns))])
    if "ed" in cfg.sources:
        new_col = np.array([True if x in [("Concentration")] else False for x in X_train.columns.droplevel(-1)])
        assert len(X_train[("Concentration")].columns) == sum(new_col)
        columns = columns | new_col

      
    elif "icp" in cfg.sources:    
        columns = np.array([True for x in range(len(X_train.columns))])

    elif "stacked" in cfg.sources:
        new_col = np.array([True if x in ["Prob"] else False for x in X_train.columns.droplevel([1,2]) ])
        columns = columns | new_col
    return columns




def cancel_out_threshold(X_train, X_test, columns, cfg):
    """
    Get rid of threshold data such as >0.05 that happens in ICP.
    If cfg.method.cancel_out_threshold is defined and true all these data points are replaced with 0. 
    Otherwise these datapoints are replaced with the value after the bigger than sign (i.e. >0.5 turns into 0.5) 
    """
    if "cancel_out_threshold" in cfg:
        if cfg.cancel_out_threshold:
            X_train.loc[:,columns] = X_train.loc[:,columns].apply(pd.to_numeric,errors="coerce").fillna(0).copy()
            X_test.loc[:,columns] = X_test.loc[:,columns].apply(pd.to_numeric,errors="coerce").fillna(0).copy()
        else:
            X_train.loc[:,columns] = X_train.loc[:,columns].replace(to_replace='<', value='',regex=True)
            X_train.loc[:,columns] = X_train.loc[:,columns].apply(pd.to_numeric,errors="coerce").fillna(0)
            X_test.loc[:,columns] = X_test.loc[:,columns].replace(to_replace='<', value='',regex=True)
            X_test.loc[:,columns] = X_test.loc[:,columns].apply(pd.to_numeric,errors="coerce").fillna(0)
    return X_train, X_test



def to_long_format(X_train, X_test, columns, idx_categorical, cfg):
    """
    Convert categorical columns into longs
    arguments:
        X_train: the training dataframe
        X_test: the test dataframe
        columns: the columns to be used
        idx_categorical: the idx of categorical columns
        cfg: the config
    """
    bool_categorical_cols = [True if x in idx_categorical else False for x in range(len(X_train.columns))]
    X_train.loc[:, bool_categorical_cols] = X_train.loc[:, bool_categorical_cols].astype('category')
    X_test.loc[:, bool_categorical_cols] = X_test.loc[:, bool_categorical_cols].astype('category')
    X_train.loc[:, bool_categorical_cols] = X_train.loc[:, bool_categorical_cols].apply(lambda x: x.cat.codes)
    X_test.loc[:, bool_categorical_cols] = X_test.loc[:, bool_categorical_cols].apply(lambda x: x.cat.codes)
    return X_train, X_test



def include_reference(X, y, data,columns, target_tuple, cfg):
    """
    If cfg.source.include_reference_data is true, reference data will be included into the training data. Otherwise is exluded.
    """
    mapper = utils.mapper(cfg.target,  cfg.stone_type, str2num=True) #From str to number (int)
    if "include_reference_data" in cfg.sources:
        if cfg.sources.include_reference_data in ([0,1]): #0 include, #1 use only it
            if cfg.sources.cancel_out_threshold:
                data.dict_df_reference.loc[:,columns] = data.dict_df_reference.loc[:,columns].apply(pd.to_numeric,errors="coerce").fillna(0).copy()
            else:
                data.dict_df_reference.loc[:,columns] = data.dict_df_reference.loc[:,columns].replace(to_replace='<', value='',regex=True)
                data.dict_df_reference.loc[:,columns] = data.dict_df_reference.loc[:,columns].apply(pd.to_numeric,errors="coerce").fillna(0).copy()
            X_ref = data.dict_df_reference.loc[:,columns].fillna(0)
            to_keep = (X_ref.T != 0).any()
            X_ref = X_ref[to_keep]
            if cfg.target  == "Origin":
                y_ref = np.array(mapper(data.dict_df_reference.loc[:,"Origin"]))
            else:
                y_ref = np.array(mapper(data.dict_df_reference.loc[:,"Origin"]))
            y_ref = y_ref[to_keep]
            if cfg.sources.include_reference_data == 0:
                X = np.concatenate([X,X_ref])
                y = np.concatenate([y,y_ref])
            elif cfg.sources.include_reference_data == 1:
                X = X_ref.values
                y = y_ref.values
            else:
                raise KeyError
    return X, y


def test_data(estimator, X_train_values, X_val_value, X_test_value):
    #Add Time as return value    
    if type(estimator) == RidgeClassifierCV:
        # if we are training on heat, d will be a 1D numpy array consisting of n elemens, where n is the number of spectras.
        # Special care is required in this case (passing d through a sigmoid), otherwise we use the softmax

        # Train values
        d = estimator.decision_function(X_train_values)
        if len(d.shape) == 1: #Sigmoid
            prob_train_class_1 = 1/(1+np.exp(-d)) 
            prob_train_class_0 = 1 - prob_train_class_1
            prob_train = np.array([prob_train_class_0, prob_train_class_1]).T  # number_of_spectras, 2 dimensions
        
        else:  #Softmax
            prob_train = np.exp(d) / np.sum(np.exp(d),axis=1)[:,None]
            
        d = estimator.decision_function(X_val_value)
        if len(d.shape) == 1: #Sigmoid
            prob_val_class_1 = 1/(1+np.exp(-d)) 
            prob_val_class_0 = 1 - prob_val_class_1
            prob_val = np.array([prob_val_class_0, prob_val_class_1]).T  # number_of_spectras, 2 dimensions
        
        else:  #Softmax
            prob_val = np.exp(d) / np.sum(np.exp(d),axis=1)[:,None]

        # Test values
        d = estimator.decision_function(X_test_value)
        if len(d.shape) == 1: #Sigmoid
            prob_test_class_1 = 1/(1+np.exp(-d)) 
            prob_test_class_0 = 1 - prob_test_class_1
            prob_test = np.array([prob_test_class_0, prob_test_class_1]).T  # number_of_spectras, 2 dimensions
        else:
            prob_test = np.exp(d) / np.sum(np.exp(d),axis=1)[:,None]
    else:
        prob_train = estimator.predict_proba(X_train_values)
        prob_val = estimator.predict_proba(X_val_value)
        prob_test = estimator.predict_proba(X_test_value)
    return prob_train, prob_val, prob_test


def validate(data, model, cfg):
    
    
    # Variables needed throughout the function
    mapper = utils.mapper(cfg.target,  cfg.stone_type, str2num=True) #From str to number 
    target_tuple = utils.target_tuple(cfg.target)

    assert len(list(cfg.sources.keys())) == 1, "Only one source is allowed for convential methods"
    source = list(cfg.sources.keys())[0]
    
        
    if len(cfg.sources.keys()) > 1:
        raise NotImplementedError

    source_key = list(cfg.sources.keys())[0]
    X_train = data.dict_df_client[source_key].loc[data.train_ids, :].copy()

    ids_train = X_train.index
    assert set(ids_train) == set(data.train_ids)

    X_test = data.dict_df_client[source_key].loc[data.test_ids, :].copy()

    ids_test = X_test.index
    assert set(X_test.index) == set(data.test_ids)

    y = np.array(mapper(data.dict_df_client["val"].loc[ids_train, target_tuple].values)) 
    y_test = np.array(mapper(data.dict_df_client["val"].loc[ids_test, target_tuple].values)) # Only used in the TabNet

    raw_y = np.array(y)
        # Convert target variable to numerical
        

    columns = pre_processing.return_columns_source(X_train,cfg)
    # idx_categorical_columns = pre_processing.return_categorical_columns(X_train, columns, cfg)
    ids_categorical_columns = []

    X_train, X_test = pre_processing.cancel_out_threshold(X_train, X_test, columns, cfg.sources[source])
    X = X_train.loc[:,columns]
    X_test = X_test.loc[:,columns]
    # X, X_test = to_long_format(X, X_test, columns, idx_categorical_columns, cfg)
    X = X.fillna(0).values.squeeze()    
    X_test = X_test.fillna(0).values.squeeze()
    
    assert all(raw_y == y[:len(raw_y)]), "Target variable is changed"


    #X_train_values = X_train.loc[:,columns].fillna(0).values.squeeze()
    X_test_value = X_test        

    prob_train, prob_test = test_data(model, X ,X_test_value,cfg)

    train_prob_dict = {}
    train_gt_dict = {}
    # Mean/Max Policy
    df_train = data.dict_df_client[source].loc[data.train_ids, :].copy()
    for ID in data.train_ids:
        curr = return_array_from_get_loc(prob_train, ID, df_train)
        if cfg.sources[source].val_method == "max":
            comp = curr.max(axis=1)
            i = np.argmax(comp)
            curr = curr[i,:]
        elif cfg.sources[source].val_method == "mean":
            comp = curr.mean(axis=0)
            curr = comp
        else: 
            raise KeyError
        train_prob_dict[ID] = curr
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        if type(gt) == str:
            train_gt_dict[ID] = gt
        else:
            train_gt_dict[ID] = gt.iloc[0] 
        #train_gt_dict[ID] = gt.pop()

    train_gt_dict = {key:mapper(y_i) for key, y_i in train_gt_dict.items()}

    df_train = learning.utils.return_pred_and_gt_pandas(train_prob_dict,train_gt_dict)

    test_prob_dict = {}
    test_gt_dict = {}
    
    df_test = data.dict_df_client[source].loc[data.test_ids, :].copy()
    for ID in data.test_ids:
        curr = return_array_from_get_loc(prob_test, ID, df_test)
        if cfg.sources[source].val_method == "max":
            comp = curr.max(axis=1)
            i = np.argmax(comp)
            curr = curr[i,:]
        elif cfg.sources[source].val_method == "mean":
            comp = curr.mean(axis=0)
            curr = comp
        else: 
            raise KeyError
        test_prob_dict[ID] = curr
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        #gt = df_ftir_test.loc[ID][target_tuple]
        if type(gt) == str:
            test_gt_dict[ID] = gt
        else:
            test_gt_dict[ID] = gt.iloc[0] 
        #test_gt_dict[ID] =  gt.pop() #We take the first element because the gt will be equal for identical IDs
    test_gt_dict = {key:mapper(y_i) for key, y_i in test_gt_dict.items()}

    df_test = learning.utils.return_pred_and_gt_pandas(test_prob_dict,test_gt_dict)
    return df_train, df_test

@utils.create_subfolder
def runner(data, cfg, counter) -> Tuple[pd.DataFrame]:
    """
    Training and returning results for a non deep learning ml algorithms (Multilayer Perceptron, Rocket, Random Forest)
    Returns:
        prob_train: df with probabilities for training data
        prob_test: df with probabilities for test data
    """
    # Variables needed throughout the function
    mapper = utils.mapper(cfg.target,  cfg.stone_type, str2num=True) #From str to number 
    target_tuple = utils.target_tuple(cfg.target)

    assert len(list(cfg.sources.keys())) == 1, "Only one source is allowed for convential methods"
    source = list(cfg.sources.keys())[0]

    source_key = list(cfg.sources.keys())[0]
    X_train = data.dict_df_client[source_key].loc[data.train_ids, :].copy()
    
  
    if cfg.method.name == "rf":
        if cfg.method.maximum_depth == "None":
            cfg.method.maximum_depth = None
        estimator = RandomForestClassifier(n_estimators=cfg.method.number_of_estimators,max_depth=cfg.method.maximum_depth, random_state=0, n_jobs=-1,class_weight=cfg.method.class_weight )

    if len(cfg.sources.keys()) > 1:
        raise NotImplementedError

    ids_train = X_train.index
    assert set(ids_train) == set(data.train_ids)

    X_test = data.dict_df_client[source_key].loc[data.test_ids, :].copy()

    assert set(X_test.index) == set(data.test_ids)

    y = np.array(mapper(data.dict_df_client["val"].loc[ids_train, target_tuple].values)) 

    raw_y = np.array(y)
        # Convert target variable to numerical
        

    columns = return_columns_source(X_train,cfg)

    #X_train, X_test = cancel_out_threshold(X_train, X_test, columns, cfg.sources[source])
    X = X_train.loc[:,columns]
    X_test = X_test.loc[:,columns]
    # X, X_test = to_long_format(X, X_test, columns, idx_categorical_columns, cfg)
    X = X.fillna(0).values.squeeze()    
    X_test = X_test.fillna(0).values.squeeze()

    assert all(raw_y == y[:len(raw_y)]), "Target variable is changed"

    if cfg.method.name == "rf":
        estimator.fit(X, y)
    else:
        raise KeyError("Unknown Method Error")

    #X_train_values = X_train.loc[:,columns].fillna(0).values.squeeze()
    X_test_value = X_test
    prob_train, prob_test = test_data(estimator, X ,X_test_value,cfg)
    train_prob_dict = {}
    train_gt_dict = {}
    # Mean/Max Policy
    df_train = data.dict_df_client[source].loc[data.train_ids, :].copy()
    for ID in data.train_ids:
        curr = return_array_from_get_loc(prob_train, ID, df_train)
        if cfg.sources[source].val_method == "max":
            comp = curr.max(axis=1)
            i = np.argmax(comp)
            curr = curr[i,:]
        elif cfg.sources[source].val_method == "mean":
            comp = curr.mean(axis=0)
            curr = comp
        else: 
            raise KeyError
        train_prob_dict[ID] = curr
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        if type(gt) == str:
            train_gt_dict[ID] = gt
        else:
            train_gt_dict[ID] = gt.iloc[0] 

    train_gt_dict = {key:mapper(y_i) for key, y_i in train_gt_dict.items()}

    df_train = learning.utils.return_pred_and_gt_pandas(train_prob_dict,train_gt_dict)

    test_prob_dict = {}
    test_gt_dict = {}
    
    df_test = data.dict_df_client[source].loc[data.test_ids, :].copy()
    for ID in data.test_ids:
        curr = return_array_from_get_loc(prob_test, ID, df_test)
        if cfg.sources[source].val_method == "max":
            comp = curr.max(axis=1)
            i = np.argmax(comp)
            curr = curr[i,:]
        elif cfg.sources[source].val_method == "mean":
            comp = curr.mean(axis=0)
            curr = comp
        else: 
            raise KeyError
        test_prob_dict[ID] = curr
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        #gt = df_ftir_test.loc[ID][target_tuple]
        if type(gt) == str:
            test_gt_dict[ID] = gt
        else:
            test_gt_dict[ID] = gt.iloc[0] 
    test_gt_dict = {key:mapper(y_i) for key, y_i in test_gt_dict.items()}

    df_test = learning.utils.return_pred_and_gt_pandas(test_prob_dict,test_gt_dict)
    return df_train, df_test  

def return_array_from_get_loc(prob, ID, df):
    """
    Returns the corresponding array in number_spectra x probability_outputs given the ID and the correspoding pandas df
    prob: probabilities from estimator output
    ID: 
    df: the dataframe from which we extract the data for computing the prob vector
    """
    res = df.index.get_loc(ID)
    if type(res) == int:
        return np.expand_dims(prob[res,:],axis=0)
    elif type(res) == np.ndarray:
        return prob[res]
    else:
        raise Error("Type not understood")
    
def load_result_data(path):
    with open(path, 'rb') as handle: 
        results = pickle.load(handle)
    return results

def unwrap_and_return(expertiment_path, counter ):
    experiment = load_result_data(expertiment_path)
    df_train = experiment.df_training.loc[counter]
    df_test = experiment.df_test.loc[counter]
    df_val = experiment.df_validation.loc[counter]
    # df_train = df_train_raw.copy()
    # df_val = df_val_raw.copy()
    # df_test = df_test_raw.copy()
    # df_train.index = df_train.index.droplevel(0)
    # df_val.index = df_val.index.droplevel(0)
    # df_test.index = df_test.index.droplevel(0)
    return df_train, df_val, df_test
    

@utils.create_subfolder
def ensemble_runner(data, cfg,counter):
    """
    Runs the ensemble method
    """
    
    mapper = utils.mapper(cfg.target,  cfg.stone_type, str2num=True) #From str to number 
    target_tuple = utils.target_tuple(cfg.target)

    # df_train_raw = {}
    # df_val_raw = {}
    # df_test_raw = {}
    df_dict_train = {}
    df_dict_val = {}
    df_dict_test = {}
    if not "uv_ensamble" in cfg.method or not cfg.method.uv_ensamble:  
        if "ed" in cfg.sources:
            #df_train_raw["ed"], df_val_raw["ed"], df_test_raw["ed"], df_dict_train["ed"], df_dict_val["ed"], df_dict_test["ed"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.ed_path))
            df_dict_train["ed"], df_dict_val["ed"], df_dict_test["ed"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.ed_path), counter)

        if "icp" in cfg.sources:    
            #df_train_raw["icp"], df_val_raw["icp"], df_test_raw["icp"], df_dict_train["icp"], df_dict_val["icp"], df_dict_test["icp"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.icp_path))
            df_dict_train["icp"], df_dict_val["icp"], df_dict_test["icp"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.icp_path), counter)
            
        if "ftir" in cfg.sources:
            #df_train_raw["ftir"], df_val_raw["ftir"], df_test_raw["ftir"], df_dict_train["ftir"], df_dict_val["ftir"], df_dict_test["ftir"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.ftir_path))
            df_dict_train["ftir"], df_dict_val["ftir"], df_dict_test["ftir"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.ftir_path), counter)
            
        if "uv" in cfg.sources:
            #df_train_raw["uv"], df_val_raw["uv"], df_test_raw["uv"], df_dict_train["uv"], df_dict_val["uv"], df_dict_test["uv"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.uv_path))
            df_dict_train["uv"], df_dict_val["uv"], df_dict_test["uv"] = unwrap_and_return(hydra.utils.to_absolute_path(cfg.method.uv_path), counter)
    
        
    # Making sure that all the val stones are not in the split
    for source in cfg.sources:
        if "_" in source:
            source = source.split("_")[0]
        training_available_stones = set(data.train_ids) & set(df_dict_train[source].index)
        assert not (training_available_stones & set(df_dict_test[source].index))
        
        assert not (set(data.test_ids) & set(df_dict_train[source].index))
    
    train_ids = set(data.train_ids)
    curr_ids = set()
    tmp = {}
    for source in cfg.sources:
        #curr_ids = curr_ids.union(df_dict_train[source].index)
        tmp[source] = df_dict_train[source]["Pred"]
        
        # Get rid of duplicates IDs
        #df_train_dict[source] = df_train_dict[source][~df_train_dict[source].index.duplicated(keep='first')]
    
    df_train = pd.concat(tmp,axis=1).sort_index()
    #df_train = df_train.loc[train_ids].sort_index()

    train_ids = sorted(list(train_ids))
    assert all(df_train.index == train_ids)
    train_ids = sorted(df_train.index)
    
    val_ids = set(data.val_ids)
    tmp = {}
    for source in cfg.sources:
        #curr_ids = val_ids.intersection(df_dict_val[source].index)
        tmp[source] = df_dict_val[source]["Pred"]
        #df_val_dict[source] = df_val_dict[source][~df_val_dict[source].index.duplicated(keep='first')]
        
    
    df_val = pd.concat(tmp,axis=1).sort_index()
    val_ids = sorted(list(val_ids))
    assert all(df_val.index == val_ids)
    val_ids = sorted(df_val.index)
    

    test_ids = set(data.test_ids)
    df_test_dict = {}
    for source in cfg.sources:
        curr_ids = test_ids.intersection(df_dict_test[source].index)
        df_test_dict[source] = df_dict_test[source].loc[curr_ids,"Pred"]
    df_test = pd.concat(df_test_dict,axis=1).sort_index()
    test_ids = sorted(df_test.index)
    assert all(df_test.index == test_ids)
    
    # Replace NaN with 0
    df_train = df_train.fillna(0)
    y_train = mapper(data.dict_df_client["val"].loc[train_ids, target_tuple])
    df_val = df_val.fillna(0)
    y_val = mapper(data.dict_df_client["val"].loc[val_ids, target_tuple])
    df_test = df_test.fillna(0)
    y_test = mapper(data.dict_df_client["val"].loc[test_ids, target_tuple])
    
    #import csem_gemintelligence
    #filtered_df_origin = csem_gemintelligence.utils.return_df_for_paper(data.dict_df_client, "origin", filter_noisy=True)
    #df_test.loc[df_test.index & filtered_df_origin["val"].index]
    
    estimator = RandomForestClassifier(n_estimators=cfg.method.number_of_estimators, random_state=1, n_jobs=-1)
    estimator.fit(df_train.values, y_train)

    prob_train, prob_val, prob_test = test_data(estimator, df_train.values, df_val.values,df_test.values)
    
    train_prob_dict = {}
    train_gt_dict = {}
    
    
    for ID in train_ids:
        curr = return_array_from_get_loc(prob_train, ID, df_train)
        assert curr.shape[0] == 1
        train_prob_dict[ID] = curr[0]
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        train_gt_dict[ID] = gt

        #train_gt_dict[ID] = gt.pop()

    train_gt_dict = {key:mapper(y_i) for key, y_i in train_gt_dict.items()}

    df_train = learning.utils.return_pred_and_gt_pandas(train_prob_dict,train_gt_dict)

    test_prob_dict = {}
    test_gt_dict = {}
    df_test = df_test
    for ID in test_ids:
        curr = return_array_from_get_loc(prob_test, ID, df_test)
        assert curr.shape[0] == 1
        test_prob_dict[ID] = curr[0]
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        test_gt_dict[ID] = gt

        #test_gt_dict[ID] =  gt.pop() #We take the first element because the gt will be equal for identical IDs
    test_gt_dict = {key:mapper(y_i) for key, y_i in test_gt_dict.items()}

    df_test = learning.utils.return_pred_and_gt_pandas(test_prob_dict,test_gt_dict)
    
    
    val_prob_dict = {}
    val_gt_dict = {}
    for ID in val_ids:
        curr = return_array_from_get_loc(prob_val, ID, df_val)
        assert curr.shape[0] == 1
        val_prob_dict[ID] = curr[0]
        gt = data.dict_df_client["val"].loc[ID, target_tuple]
        val_gt_dict[ID] = gt

        #test_gt_dict[ID] =  gt.pop() #We take the first element because the gt will be equal for identical IDs
    val_gt_dict = {key:mapper(y_i) for key, y_i in val_gt_dict.items()}

    df_val = learning.utils.return_pred_and_gt_pandas(val_prob_dict,val_gt_dict)
    


    return df_train, df_val, df_test
    
    
        
    
