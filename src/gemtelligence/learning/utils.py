from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Any, List, Tuple
import pandas as pd
from dataclasses import dataclass
import sklearn
from scipy import stats
import copy
from . import hol_net_v1
from . import training
import torch
from sklearn.metrics import log_loss, accuracy_score
from ..results import Results, ResultsCV
from sklearn.preprocessing import LabelEncoder
import hydra
from .. import utils
from pathlib import Path
import random
from . import scikit_learn_interface
from copy import deepcopy
from omegaconf import OmegaConf
import json


@dataclass
class data_class:
    df: pd.DataFrame
    main_features: List
    all_features: List
    train_ids: List = None
    val_ids: List = None
    test_ids: List = None


@dataclass
class IntrusiveClass:
    all_ids: List
    to_keep_out: List


@dataclass
class stones_class:
    X_client: np.array
    y_client: np.array
    X_reference: np.array
    y_reference: np.array


class ReadSplit:
    def __init__(self, splits) -> None:
        self.splits = splits

    def split(self, ids_fin, _):
        for i in range(len(self.splits)):
            available = set(self.splits[str(i)]["train"]) | set(
                self.splits[str(i)]["val"]
            )
            train_ids = list(set(available) & set(ids_fin))
            test_ids = list(set(self.splits[str(i)]["test"]) & set(ids_fin))
            yield np.array(train_ids), np.array(test_ids)


class CVRes:
    def __init__(
        self,
        y_train=None,
        y_val=None,
        y_test=None,
        prob_train=None,
        prob_val=None,
        prob_test=None,
        cfg=None,
    ):
        prob_train = prob_train
        prob_val = prob_val
        prob_test = prob_test

        if prob_train is not None:
            self.train_accuracy = accuracy_score(y_train, np.argmax(prob_train, axis=1))
            print(f"Train Accuracy in CVRes {self.train_accuracy}")
            self.train_cross_entropy = log_loss(
                y_train,
                prob_train,
                labels=[x for x in utils.target_list(cfg.target, cfg.stone_type)],
            )

        if prob_val is not None:
            self.val_accuracy = accuracy_score(y_val, np.argmax(prob_val, axis=1))
            print(f"Val Accuracy in CVRes {self.val_accuracy}")
            self.val_cross_entropy = log_loss(
                y_val,
                prob_val,
                labels=[x for x in utils.target_list(cfg.target, cfg.stone_type)],
            )

        if prob_test is not None:
            self.test_accuracy = accuracy_score(y_test, np.argmax(prob_test, axis=1))
            print(f"Test Accuracy in CVRes {self.test_accuracy}")
            self.test_cross_entropy = log_loss(
                y_test,
                prob_test,
                labels=[x for x in utils.target_list(cfg.target, cfg.stone_type)],
            )
        else:
            self.test_accuracy = None
            self.test_cross_entropy = None

        # if prob_test is not None:

        # else:

    def __repr__(self):
        return "Accuracy {} and {}, Cross entropy {} and {}".format(
            self.train_accuracy,
            self.val_accuracy,
            self.test_accuracy,
            self.train_cross_entropy,
            self.val_cross_entropy,
            self.test_cross_entropy,
        )


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    model.load_state_dict(
        torch.load(
            hydra.utils.to_absolute_path("./pretrained/initial.pth"),
            map_location=lambda storage, loc: storage,
        )
    )


def get_label_encoder(data, target):
    label_enc = LabelEncoder()
    if len(data.dict_df_client.columns.values[0]) == 2:
        y_test = label_enc.fit_transform(
            list(data.dict_df_client[(target, "")].cat.categories)
        )
        assert np.array_equal(y_test, np.array([0, 1, 2, 3][: len(y_test)]))
        assert np.array_equal(
            label_enc.transform(data.dict_df_client[(target, "")]),
            data.dict_df_client[(target, "")].cat.codes.values,
        )
    else:
        y_test = label_enc.fit_transform(
            list(data.dict_df_client[(target, "", "")].cat.categories)
        )
        assert np.array_equal(y_test, np.array([0, 1, 2, 3][: len(y_test)]))
        assert np.array_equal(
            label_enc.transform(data.dict_df_client[(target, "", "")]),
            data.dict_df_client[(target, "", "")].cat.codes.values,
        )

    return label_enc


def obtain_all_stones_available_in_sources(dict_df, keys=["uv", "ftir", "ed", "icp"]):
    """
    Iterate over all keys in the dict_df and return the union of the IDs stones
    returns:
        all_stones_available: set of all stones ID available
        keys: list of keys in the dict_df to consider
    """
    all_stones = []
    for key in keys:
        if key != "val":
            all_stones.extend(dict_df[key].index.values)
    return set(all_stones)


def loc_stones(curr_dict, stones_to_keep):
    """
    Remove stones that are not in stones_to_keep from each entry of the curr_dict
    """
    curr_dict = curr_dict.copy()
    for key in curr_dict.keys():
        # Pandas complains if we try to remove an ID that does not exist.
        # So first we need the intersection of the stones available in the sources and the stones to keep

        curr_df_stones_to_keep = set(stones_to_keep) & set(curr_dict[key].index)
        curr_dict[key] = curr_dict[key].loc[curr_df_stones_to_keep]
    return curr_dict


def group_rechecks(df, stone_type, validation_data=[]):
    same_stones = set()
    rechecks = df["val"]["Rechecks"]
    for id, value in rechecks.iteritems():
        assert id in value
        if len(value) > 1:
            same_stones.add(tuple(value))

    total_numer_of_duplicate_stones = sum([len(x) for x in same_stones]) - len(
        same_stones
    )
    print("Total number of children stones {}".format(total_numer_of_duplicate_stones))

    new_old_dict = {}
    for stones in same_stones:
        # Get the youngest stone. It will be the parent
        youngest_stone = max(stones)
        new_old_dict[youngest_stone] = [x for x in stones if x != youngest_stone]
    # Change the Heat Treament of the New stone only if it is None and before was not None
    if stone_type != "emerald":
        target = "Heat Treatment Value"
    else:
        target = "Clarity Enhancement Value"

    for new, olds in new_old_dict.items():
        new_origin = df["val"].at[new, target]
        if (new_origin is None) or (new_origin == "None"):
            for old in sorted(olds, reverse=True):
                if (new_origin is None) or (new_origin == "None"):
                    old_origin = df["val"].at[old, target]
                    if not ((old_origin is None) or (old_origin == "None")):
                        df["val"].at[new, target] = old_origin
                        break

    # assign all sources of older stone to current stone
    for new, olds in new_old_dict.items():
        for source in df.keys() - ["val"]:
            for old in olds:
                if old in df[source].index and not old in validation_data:
                    # if source == "icp":
                    # breakpoint()
                    df[source].rename(index={old: new}, inplace=True)
                    assert not old in df[source].index

    return df


def pre_query_correct_stones(curr_dict, cfg):
    """
    Set up val entry of the curr_dict with the correct stones (i.e. stones that have the target in well specified category) given the stone_type and target
    """
    curr_df = curr_dict["val"]

    if cfg.target == "origin":
        curr_df, _ = utils.filter_countries(curr_df, stone_type=cfg.stone_type)
    # Preparing heat treatment data
    elif cfg.target == "treatment":
        curr_df, _ = utils.filter_heat_treatment(curr_df, stone_type=cfg.stone_type)
    elif cfg.target == "pre_origin":
        curr_df, _ = utils.filter_basaltic_vs_basaltic(
            curr_df, stone_type=cfg.stone_type
        )
    else:
        raise KeyError("Target {} not supported".format(cfg.target))

    if cfg.experiments.train_only_on_country:
        # Keep only the stones that are in the country of the target
        curr_df, _ = (
            curr_df.loc[
                curr_df["ExcelOrigin"] == cfg.experiments.train_only_on_country
            ],
            [],
        )

    # This is used when we want to train together for heat and origin
    if cfg.add_secondary_target:
        if cfg.target == "origin":
            curr_df = utils.rename_heat_treatment_columns(curr_df)
            bool_cond = ~curr_df["Heat Treatment Value"].isin(["NTE", "TE"])
            curr_df.loc[bool_cond, "Heat Treatment Value"] = -1
            curr_df["Heat Treatment Value"] = curr_df["Heat Treatment Value"].astype(
                "category"
            )
        elif cfg.target == "treatment":
            bool_cond = ~curr_df["FinalOrigin"].isin(
                utils.target_list("origin", cfg.stone_type)
            )
            curr_df.loc[bool_cond, "FinalOrigin"] = -1

    # Drop unsued categories
    curr_dict["val"] = curr_df

    # Drop duplicate indices
    curr_dict["val"] = curr_dict["val"].loc[
        ~curr_dict["val"].index.duplicated(keep="first")
    ]

    stones_to_keep = set(curr_df.index)
    curr_dict = loc_stones(curr_dict, stones_to_keep)
    return curr_dict


def post_query_correct_stones(curr_dict, key):
    """
    Keep only in the val entry of the curr_dict the stones that appear at least once in the data sources considered
    """
    stones_id = obtain_all_stones_available_in_sources(curr_dict, key)
    print(f"Total number of stones available {len(stones_id)}")
    curr_dict["val"] = curr_dict["val"].loc[stones_id]
    return curr_dict


def real_validation_stones(data, stones):
    """Returns validation stones together with their recheck counterparts

    Args:
        data (_type_): _description_
        stones (_type_): _description_

    Returns:
        _type_: _description_
    """
    same_stones = set()
    rechecks = data["val"]["Rechecks"]
    for id, value in rechecks.iteritems():
        assert id in value
        if id in stones:
            same_stones.add(tuple(value))
    return [x for group in same_stones for x in group]


def remove_validation_stones(data, stones):
    for stone in stones:
        for source in data.keys():
            if stone in data[source].index:
                data[source].drop(stone, axis=0, inplace=True)
            assert not stone in data[source].index
    return data


def final_postprocessing(dict_df_client, cfg):
    """
    Postprocessing of the results.
    """
    for key in dict_df_client.keys():
        if key in ["ed", "icp"]:
            dict_df_client[key] = dict_df_client[key].apply(
                pd.to_numeric, errors="coerce"
            )
            for col in dict_df_client[key].columns:
                min_val = dict_df_client[key][col].min()
                diff = (dict_df_client[key][col].mean() - min_val) * 0.5
                to_be_replaced = min(0, min_val - diff)

                # Fill the NaN values with the minimum value
                dict_df_client[key][col].fillna(to_be_replaced, inplace=True)

    return dict_df_client


def cv_training(
    cross_val_method, data: data_class, cfg, split_val_test=True
) -> Results:
    class_len = len(utils.target_list(cfg.target, cfg.stone_type))
    # TODO: Move this in data validation
    tmp = data.dict_df_client["icp"].apply(pd.to_numeric, errors="coerce")
    for col in tmp.columns:
        tmp = tmp.loc[~(tmp[col] < 0)]
    tmp = tmp.loc[~(tmp["Al27"] < 500000)]
    tmp = tmp.loc[~(tmp["Nd146"] > 10)]
    data.dict_df_client["icp"] = tmp

    cfg.secondary_class_len = 0
    cfg.class_len = class_len  # We had this information so we can use it in the model
    # Get the correct IDs. See docstring of each func for more details
    if cfg.is_consistent_exp:
        data_to_feed = copy.deepcopy(data)

    data.dict_df_client = group_rechecks(data.dict_df_client, cfg.stone_type)
    data.dict_df_client = pre_query_correct_stones(data.dict_df_client, cfg)
    data.dict_df_client = post_query_correct_stones(
        data.dict_df_client, list(cfg.sources.keys())
    )
    data.dict_df_client = final_postprocessing(data.dict_df_client, cfg)
    if not cfg.is_consistent_exp:
        data_to_feed = data
    # This should pass, since this stone was rechecked (19127020)
    assert not 13086040 in data.dict_df_client["val"].index
    assert not 16030055 in data.dict_df_client["val"].index  # First stone of a recheck

    curr_dict = deepcopy(
        data.dict_df_client
    )  # This is the dict that will be used for training/validation
    # Small log about the number of stones available

    res = ResultsCV(data_to_feed, cfg)
    # df = data.dict_df.copy()

    # Converting the target column to integers (We are using our own funcs for this because Pandas is unreliable)
    mapper = utils.mapper(cfg.target, cfg.stone_type, str2num=True)  # Str to num
    target_tuple = utils.target_tuple(cfg.target)
    val_dict = curr_dict["val"]
    val_dict[target_tuple] = val_dict[target_tuple].apply(mapper)

    y_ids_mapping = val_dict
    assert len(y_ids_mapping.index) == len(set(y_ids_mapping.index))
    ids = y_ids_mapping.index.values

    ids_fin = val_dict.loc[ids, target_tuple].index
    ori_fin = val_dict.loc[ids, target_tuple].values
    # Assert that ids_fin are all unique
    assert len(ids_fin) == len(set(ids_fin))

    # Only for generating train/val ids
    if not cfg.use_fix_generator_path_val:  # cfg.use_fix_generator_path:
        splits = {}
        for iii, indices in enumerate(cross_val_method.split(ids_fin, ori_fin)):
            train_index, test_index = indices
            train_ids = set(np.array(ids_fin)[train_index])
            val_test_ids = set(np.array(ids_fin)[test_index])
            splits[iii] = {
                "train": [int(x) for x in train_ids],
                "test": [int(x) for x in val_test_ids],
            }
        # with open("splits.json", "w") as f:
        #     import json
        #     json.dump(splits, f)
    else:
        p = hydra.utils.to_absolute_path(cfg.use_fix_generator_path_val)
        import json

        with open(p, "r") as f:
            splits = json.load(f)

            # yield self.splits[]
        tmp = list(cross_val_method.split(ids_fin, ori_fin))[0][0]
        before = np.array(ids_fin)[tmp]
        cross_val_method = ReadSplit(splits)
        if set(cfg.sources.keys()) == {"ftir", "icp", "uv", "ed"}:

            after = list(cross_val_method.split(ids_fin, ori_fin))[0][0]
            assert [x == y for x, y in zip(sorted(before), sorted(after))]
        # Create a generator for the train/val ids

    counter = 0
    splits = {}

    # Compute statistics over the amout of data available for each method
    total_number_of_stones = len(ids_fin)
    print("Total number of stones {}".format(total_number_of_stones))
    total_stones_per_method = {}
    all_method = set(ids_fin)
    for source in cfg.sources.keys():
        total_stones_per_method[source] = set(curr_dict[source].index) & set(ids_fin)
        all_method = all_method & total_stones_per_method[source]
        print(
            "Total number of stones for {} {}".format(
                source, len(total_stones_per_method[source])
            )
        )

    for train_index, test_index in cross_val_method.split(ids_fin, ori_fin):
        print("CV split {}".format(counter + 1))
        if cfg.use_fix_generator_path_val:
            train_ids = train_index
            test_ids = test_index
        else:
            train_ids = np.array(ids_fin)[train_index]
            test_ids = np.array(ids_fin)[test_index]

        if split_val_test:
            ori_train = val_dict.loc[train_ids][target_tuple]
            train_ids, val_ids, _, _ = train_test_split(
                ori_train.index,
                ori_train.values,
                stratify=ori_train,
                shuffle=True,
                random_state=2,
                test_size=0.25,
            )

            if not cfg.use_fix_generator_path_val:
                splits[counter] = {
                    "train": [int(x) for x in train_ids],
                    "val": [int(x) for x in val_ids],
                    "test": [int(x) for x in test_ids],
                }

                if counter == 4:
                    import json

                    # Save the splits
                    with open("splits_val.json", "w") as f:
                        json.dump(splits, f)
            else:
                import json

                p = hydra.utils.to_absolute_path(cfg.use_fix_generator_path_val)
                with open(p, "r") as f:
                    splits = json.load(f)

                train_ids_candidate = set(splits[str(counter)]["train"]) & set(ids_fin)
                val_ids_candidate = set(splits[str(counter)]["val"]) & set(ids_fin)

                test_ids_candidate = set(splits[str(counter)]["test"]) & set(ids_fin)
                if set(cfg.sources.keys()) == {"ftir", "icp", "uv", "ed"}:
                    assert [
                        x == y
                        for x, y in zip(sorted(train_ids), sorted(train_ids_candidate))
                    ]
                    assert [
                        x == y
                        for x, y in zip(sorted(val_ids), sorted(val_ids_candidate))
                    ]
                    assert [
                        x == y
                        for x, y in zip(sorted(test_ids), sorted(test_ids_candidate))
                    ]

                train_ids = train_ids_candidate
                val_ids = val_ids_candidate
                test_ids = test_ids_candidate

            not_train_ids = list(val_ids) + list(test_ids)
            assert not len(set(val_ids) & set(test_ids))
            assert not len(set(train_ids) & set(val_ids))
            assert not len(set(train_ids) & set(test_ids))

            # train_ids = train_ids #train_ids[train_ids]
            # test_ids = test_ids
        else:
            val_ids = val_test_ids
            test_ids = val_test_ids
            assert len(val_ids) + len(train_ids) == len(ids)
            assert set(val_ids) & set(train_ids) == set()

        tmp = set(train_ids) & set(
            val_dict.loc[val_dict[("Considered_Training")] == True].index
        )  # - set(ori_val_test.index)
        if cfg.filter_years == True:
            tmp = [x for x in list(tmp) if str(x)[:2] in ["17", "18", "19", "21", "20"]]
        train_bool = [True if x in tmp else False for x in val_dict.index]
        assert sum(train_bool) == len(tmp)
        D_train = val_dict.loc[tmp]  # ,# imp_col]
        D_val = val_dict.loc[val_ids]  # , imp_col]
        D_test = val_dict.loc[test_ids]  # , imp_col]
        data.train_ids, data.val_ids, data.test_ids = (
            D_train.index,
            set(val_ids),
            set(test_ids),
        )
        # X_train, X_val, X_test = D_train, D_val, D_test
        y_train, y_val, y_test = (
            D_train[target_tuple],
            D_val[target_tuple],
            D_test[target_tuple],
        )
        assert set(y_train.index) & set(y_test.index) == set()
        if cfg.method.model_type == "torch":
            prob_train, prob_val, prob_test = training.runner(data, cfg, counter)
        elif cfg.method.model_type == "traditional" and cfg.method.name != "ensemble":
            # Traditional method uses sciki-learn (Used for the publication only)
            prob_train, prob_test = scikit_learn_interface.runner(data, cfg, counter)
        elif cfg.method.model_type == "traditional" and (
            cfg.method.name == "ensemble" or cfg.method.name == "linearlayer"
        ):
            # Traditional method uses sciki-learn (Used for the publication only)
            prob_train, prob_val, prob_test = scikit_learn_interface.ensemble_runner(
                data, cfg, counter
            )

        if not cfg.is_consistent_exp:
            assert set(y_train.index) == set(prob_train.index)
            assert set(y_test.index) == set(prob_test.index)

            assert all(prob_test.loc[test_ids].iloc[:, -1].values == y_test.values)

            # Sort prob_train, prob_test and y_train and y_test by index

            prob_train = prob_train.sort_index()
            y_train = y_train.sort_index()

            assert all(prob_train.iloc[:, -1] == y_train)

            prob_train_np = prob_train.iloc[:, :class_len].values

            if prob_val is not None:
                prob_val = prob_val.sort_index()
                y_val = y_val.sort_index()
                assert all(prob_val.iloc[:, -1] == y_val)
                prob_val_np = prob_val.iloc[:, :class_len].values
        else:
            prob_val_np = None
            prob_train_np = None
        if prob_test is not None:
            prob_test = prob_test.sort_index()
            y_test = y_test.sort_index()
            prob_test_np = prob_test.iloc[:, :class_len].values
            if not cfg.is_consistent_exp:
                assert all(prob_test.iloc[:, -1] == y_test)
            else:
                y_test = prob_test.iloc[:, -1]
                test_ids = prob_test.index

        curr = CVRes(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            prob_train=prob_train_np,
            prob_val=prob_val_np,
            prob_test=prob_test_np,
            cfg=cfg,
        )

        # validation_data = sorted(validation_data)
        train_ids = sorted(list(train_ids))
        val_ids = sorted(list(val_ids))
        test_ids = sorted(list(test_ids))

        if prob_train is not None:
            res.fill_results(train_ids, prob_train_np, mode="train")

        if prob_val is not None:
            res.fill_results(val_ids, prob_val_np, mode="val")
        if prob_test is not None:
            res.fill_results(test_ids, prob_test_np, mode="test")
        res.update_cv_res(curr)

        counter = counter + 1

    res.close_results()

    return res


def y_mapping_creator(y_mapping):
    def m(x):
        return y_mapping[x]

    return m


def normalize_pandas(df, columns):
    df.reset_index(inplace=True, drop=True)
    X_client_scaled = sklearn.preprocessing.MinMaxScaler().fit_transform(
        df.loc[:, columns]
    )
    df.loc[:, columns] = pd.DataFrame(X_client_scaled, columns=columns)
    return df


def remove_outliers(df, columns=[], z_score_threshold=4):
    df = copy.deepcopy(df)
    if not len(columns):
        return df[(np.abs(stats.zscore(df)) < z_score_threshold).all(axis=1)]
    else:
        return df[(np.abs(stats.zscore(df[columns])) < z_score_threshold).all(axis=1)]


def decision(probability):
    return random.random() < probability


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def return_pred_and_gt_pandas(dict_fin, dict_gt) -> pd.DataFrame:
    """
    Args:
        dict_fin: A ID to probability dictionary
        dict_gt: A ID to gt dictionary
    Returns:
        A pandas series combining the two dictionaries
    """

    df = pd.DataFrame.from_dict(dict_fin, orient="index")
    df_y_gt = pd.DataFrame.from_dict(dict_gt, orient="index")
    df_fin = pd.concat([df, df_y_gt], axis=1)
    return df_fin
