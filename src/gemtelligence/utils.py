from hydra import utils
import regex as re
import pdb
import numpy as np
from scipy.interpolate import interp1d
import os

# from collections import namedtuple, Counter
import pandas as pd

# from typing import Dict, List, Union
import pathlib
from geopy.geocoders import Nominatim
import pdb
import math
import warnings
import shutil
import hydra
from collections.abc import Iterable
from functools import partial
import glob
from pathlib import Path
import json
from copy import deepcopy
from collections import defaultdict, Counter
import pickle

warnings.filterwarnings("default", module="csem_gemintelligence.utils")


# Helper functions
def filter_data_for_task(raw_data, task, years=None):
    filtered_data = utils.return_df_for_paper(raw_data, task, filter_noisy=True, years=years)
    available_ids = filtered_data["val"].index
    return filtered_data, available_ids


def save_json(data, path):
    pathlib.Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def places_to_coordinates(locations):
    geolocator = Nominatim(user_agent="tommaso.bendinelli@csem.ch")
    try:
        res = list(map(geolocator.geocode, locations))
        coord = list(map(lambda x: x[1], res))
    except:
        pdb.set_trace()
    return coord


def manual_fix_names(locations):
    first = locations.replace("Madagaskar", "Madagascar")
    second = first.replace("Kashmir (customer stone)", "Kashmir")
    warnings.warn("Fix Andranandambo location")
    third = second.replace("Madagascar (Andranandambo)", "Madagascar")
    fourth = third.replace("Sri Lanka (Okkampitaya)", "Sri Lanka (Okkampitiya)")
    return fourth


def HT_mapper(x):
    try:
        d = {"NTE": 0, "NTE > TE": 0.25, "TE": 1, "TE > NTE": 0.75, "TE1 low": 0.75}
        return d[x]
    except:
        return x


def ST_mapper(x):
    try:
        d = {
            "medium": 0.5,
            "medium to saturated": 0.75,
            "saturated": 1,
            "strongly saturated": 1.25,
            "weak": 0,
            "weak to medium": 0.25,
            "weak to saturated": 0.5,
        }
        return d[x]
    except:
        return x


def Tone_mapper(x):
    try:
        d = {
            "dark": 1,
            "light": 0,
            "light to moderate": 0.5,
            "light to moderate/dark": 0.75,
            "moderate": 0.75,
            "moderate to dark": 1,
            "very light to dark": 0.5,
            "very dark": 1.25,
        }
        return d[x]
    except:
        return x


def remapping_args_dict(x):
    """
    Dummy function to return remapping dictionary (from args to df)
    """
    remapping_dict = {
        "ftir": "FTIR_DATA",
        "val": "master_file",
        "ed": "ED_DATA",
        "icp": "ICP_DATA",
        "uv": "UV_DATA",
        "nlp": "nlp",
        "image": "image",
    }
    return remapping_dict[x]


def remove_precise_location(string):
    if string == None:
        return
    if string == "Azad Kashmir":
        return "Kashmir"
    if string in ["Zambia (Musakashi)", "Zambia (Kafubu)"]:
        return "Zambia"

    if type(string) == float and math.isnan(string):
        return "not determinable"
    if string is None:
        return "not determinable"
    try:
        regex = re.compile("^[^(]*")
        result = re.findall(regex, string)
    except TypeError:
        pdb.set_trace()
        raise Exception
    return result[0].strip()


def remove_spaces_columns_name(edxrf_compact_name):
    edxrf_compact_name.columns = [
        col.strip() for col in edxrf_compact_name.columns.values
    ]
    return edxrf_compact_name


def return_best_estimator_from_sklearn(res: dict):
    max_index = res["test_score"].argmax()
    best_estimator = res["estimator"][max_index]
    return best_estimator


def reg_extract(string, reg):
    regex = re.compile(reg)
    try:
        title = re.search(regex, string).group(0)
    except:
        raise RuntimeError("Cannot match", string)
    return title


def reg_filter_list(li, reg):
    regex = re.compile(reg)
    selected_files = list(filter(lambda x: regex.search(x), li))
    return selected_files


def reg_filter(li, reg):
    regex = re.compile(reg)
    selected_files = list(filter(lambda x: regex.search(x.name), li))
    return selected_files


def reg_selector(s, reg):
    regex = re.compile(reg)
    part = regex.findall(s)[0]
    return part


def reg_return_groups(string, reg):
    regex = re.compile(reg)
    res = ["".join(i).strip() for i in regex.findall(string)]
    return res


def return_row_correct_unit_of_measure(s):
    """Data is converted to standard unit of measure"""
    for col, val in zip(s["Concentration"].index, s["Concentration"]):
        if "Std.Dev" in s.index:
            breakpoint()
            print("Std.Dev should not be in the dataframe anymore")
            if type(s["Std.Dev"][col]) == str:
                # dropping wrong ed data by setting ID to nan and later deleting said rows
                s.loc[:] = np.nan
                break

            if type(val) == float:
                continue  # s["Concentration"][col] = val
            elif type(val) != str:
                s.loc["Concentration", col] = 0
                s.loc["Std.Dev", col] = 0
            else:
                regex = re.compile(r"\d*\.?\d+")
                num = float(regex.findall(val)[0])
                if "ppm" in val:
                    s.loc["Concentration", col] = float(num * ((10) ** (-6)))
                    s.loc["Std.Dev", col] = float(s["Std.Dev"][col] * ((10) ** (-6)))
                elif "%" in val:
                    s.loc["Concentration", col] = float(num * ((10) ** (-2)))
                    s.loc["Std.Dev", col] = float(s["Std.Dev"][col] * ((10) ** (-2)))
                else:
                    # dropping wrong ed data
                    s.loc[:] = np.nan
        else:
            if type(val) == float:
                continue  # s["Concentration"][col] = val
            elif type(val) != str:
                s.loc["Concentration", col] = 0
            else:
                regex = re.compile(r"\d*\.?\d+")
                num = float(regex.findall(val)[0])
                if "ppm" in val:
                    s.loc["Concentration", col] = float(num * ((10) ** (-6)))
                elif "%" in val:
                    s.loc["Concentration", col] = float(num * ((10) ** (-2)))
                else:
                    # dropping wrong ed data
                    s.loc[:] = np.nan

    return s


def create_subfolder(func):
    """Decorator function that creates the subfolder for 'func' (=which is 'runner' or 'runner_reference')"""

    def wrapper(*args):
        pathlib.Path(str(args[-1])).mkdir(exist_ok=True)
        if not ".hydra" in os.listdir():
            shutil.copytree(".hydra", os.path.join(str(args[-1]), ".hydra"))
        os.chdir(str(args[-1]))
        res = func(*args)
        os.chdir("../")
        return res

    return wrapper


def load_h5(path, raw_key):
    """
    Loads a h5 file and returns the dataframe corresponding to the key
    """
    key = remapping_args_dict(raw_key)
    df = pd.read_hdf(path, key=key, mode="r")
    # if raw_key == "ed":
    #     # TODO: remove this when ED is fixed
    #     df.index = df["index"]
    return df


def load_unique_specific_source_client_data(stone_type, source=None):
    """
    Load specific data source from the client data (from the folder stone_type/final/)
    For rubies we are saving our data into an h5 file.
    args:
       stone_type: string, either "sapphire", "ruby" or "emerald"
       source:  .
    """

    client_data = {}
    val = load_h5(
        hydra.utils.to_absolute_path(f"data/automatic/{stone_type}/master_file.h5"),
        "val",
    )
    client_data["val"] = val

    # Making ruby data compliant with the standards
    # data_dfs = []
    if source:
        df = load_h5(
            hydra.utils.to_absolute_path("data/automatic/ruby/master_file.h5"), source
        )
        client_data[source] = df

    return client_data


def load_client_data(stone_type="sapphire") -> dict:
    """
    Loads client data from the data automatic folder depending on the stone type.
    args:
        stone_type: string, either "sapphire", "ruby" or "emerald"
        ignore_loading_entries: list of strings key, they represent the columns of the dataframe that should be ignored.
    returns:
        client_data: a dictionary of dataframes, each dataframe is a different source.
    """
    client_data = {}
    sources = ["uv", "ftir", "ed", "icp", "image", "val", "nlp"]
    path = hydra.utils.to_absolute_path(f"data/master_file.h5")
    for source in sources:
        try:
            df = load_h5(path, source)
        except Exception as e:
            print(e)
            print(f"Warning no source {source}")
            df = pd.DataFrame()
        client_data[source] = df

    if all(len(client_data[x]) == 0 for x in sources):
        raise Exception("Source file not found!")

    return client_data


def load_sapphires_stones_from_path(path="data/master_file_reference.h5") -> dict:
    reference_data = {}
    sources = ["uv", "ftir", "ed", "icp", "image", "val", "nlp"]
    path = hydra.utils.to_absolute_path(path)
    for source in sources:
        try:
            df = load_h5(path, source)
        except Exception as e:
            print(e)
            print(f"Warning no source {source}")
            df = pd.DataFrame()
        reference_data[source] = df

    if all(len(reference_data[x]) == 0 for x in sources):
        raise Exception("Source file not found!")

    return reference_data


def return_sapphire_image_path():
    """
    Dummy function to return sapphire image path
    """
    return hydra.utils.to_absolute_path("/home/mak/data/images")


def return_ruby_image_path():
    """
    Dummy function to return ruby image path
    """
    return hydra.utils.to_absolute_path(
        "/data_gem/raw_data/ruby/client_stones/Ruby_Inclusion_Pics"
    )


def return_emerald_image_path():
    """
    Dummy function to return emerald image path
    """
    return hydra.utils.to_absolute_path(
        "/data_gem/raw_data/emerald/client_stones/images/Emerald_Inclusion_Pics"
    )


sapphire_countries = ["Burma", "Kashmir", "Madagascar", "Sri Lanka"]
pre_classifier = ["Big4", "other"]
ruby_countries = [
    "Burma",
    "Thailand",
    "Mozambique",
    "Afghanistan",
    "Madagascar",
    "Kenya",
    "Vietnam",
    "Tajikistan",
    "Tanzania",
    "East Africa",
]
emerald_countries = [
    "Colombia",
    "Zambia",
    "Afghanistan",
    "Brazil",
    "Russia",
    "Ethiopia",
    "East Africa",
    "Madagascar",
    "Zimbabwe",
]
heat_conditions = ["NTE", "TE"]
clarity_conditions = ["NCE", "CE"]


def target_list(target, stone_type):
    """
    Returns a list containg all the classes based on target_type and stone_type
    """
    if target == "origin":
        if stone_type == "sapphire":
            return sapphire_countries
        else:
            raise KeyError
    elif target == "treatment":
        return heat_conditions
    elif target == "pre_origin":
        return pre_classifier
    else:
        raise KeyError


def mapper(target, stone_type, str2num):
    """
    Returns a given mapper function depending on the stone type and target.
    If str2num is True, the mapper function will accept a string and return a number.
    If str2num is False, the mapper function will accept a number and return a string.
    """
    if str2num:
        if target == "origin":
            return partial(country_mapper, stone_type=stone_type)
        elif target == "treatment":
            return heat_mapper
        elif target == "pre_origin":
            return partial(
                country_mapper, classification_type="pre_origin", stone_type=stone_type
            )
        else:
            raise KeyError("Target not recognized")
    else:
        if target == "origin":
            return partial(inv_country_mapper, stone_type=stone_type)
        elif target == "treatment":
            return inv_heat_mapper
        elif target == "pre_origin":
            return partial(
                inv_country_mapper,
                classification_type="pre_origin",
                stone_type=stone_type,
            )
        else:
            raise KeyError("Target not recognized")


def clarity_mapper(y, error="raise"):
    """
    Mapping from clarity conditions to a number.
        If y is in the list of clarity conditions, then return the index of the list.
        If y is a list, then return a list of indices.
        If error is set to "raise", then an error will be raised if the country is not in the list of countries.
        If error is set to "coerce", then the function will return a nan.
    """
    c = {i: k for k, i in enumerate(clarity_conditions)}
    if type(y) == str:
        d = c[y]
    elif isinstance(y, Iterable):
        d = [c[i] for i in y]
    else:
        if error == "raise":
            raise ValueError("Heat condition not recognized")
        elif error == "coerce":
            return np.nan
    return d


def inv_clarity_mapper(y):
    """
    Mapping from a number to a clarity condition.
       If y is a int, then returns the corresponding clarity condition.
       If y is a listof int, then returns a list of corresponding clarity conditions.
    """
    c = clarity_conditions
    if type(y) == int:
        d = c[y]
    elif isinstance(y, Iterable):
        d = [c[i] for i in y]
    else:
        raise KeyError
    return d


def heat_mapper(y, error="raise"):
    """
    Mapping from heat conditions to a number.
        If y is in the list of heat conditions, then return the index of the list.
        If y is a list, then return a list of indices.
        If error is set to "raise", then an error will be raised if the country is not in the list of countries.
        If error is set to "coerce", then the function will return a nan.
    """
    c = {i: k for k, i in enumerate(heat_conditions)}
    if type(y) == str:
        d = c[y]
    elif isinstance(y, Iterable):
        d = [c[i] for i in y]
    else:
        if error == "raise":
            raise ValueError("Heat condition not recognized")
        elif error == "coerce":
            return np.nan
    return d


def inv_heat_mapper(y):
    """
    Mapping from a number to a heat condition.
       If y is a int, then returns the corresponding heat condition.
       If y is a listof int, then returns a list of corresponding heat conditions.
    """
    c = heat_conditions
    if type(y) == int:
        d = c[y]
    elif isinstance(y, Iterable):
        d = [c[i] for i in y]
    else:
        raise KeyError
    return d


def country_mapper(y, stone_type, classification_type="big_four", error="raise"):
    """
    Mapping from countries to value
        If input is a country, the function will return a int
        If input is a list, the function will rerturn a list of int
        If error is set to "raise", then an error will be raised if the country is not in the list of countries.
        If error is set to "coerce", then the function will return a nan.
    """
    if classification_type == "big_four":
        c = target_list("origin", stone_type)
    elif classification_type == "pre_origin":
        c = target_list("pre_origin", stone_type)
    else:
        raise KeyError("Classification type not recognized")
    c = {i: k for k, i in enumerate(c)}
    if type(y) == str:
        if y in c:
            d = c[y]
        else:
            if error == "raise":
                raise ValueError(f"Country {y} not recognized")
            elif error == "coerce":
                return np.nan
    elif isinstance(y, Iterable):
        for i in y:
            if i not in c:
                if error == "raise":
                    raise ValueError(f"Country {i} not recognized")
                elif error == "coerce":
                    return np.nan
        d = [c[i] for i in y]
    return d


def inv_country_mapper(y, stone_type, classification_type="big_four"):
    """Mapping from value to countries
    If input is a int, the function will return a str
    If input is an iterable, the function will return a list of str
    """
    if classification_type == "big_four":
        c = target_list("origin", stone_type)
    elif classification_type == "pre_origin":
        c = target_list("pre_origin", stone_type)
    else:
        raise KeyError("Classification type not recognized")
    if type(y) == int:
        d = c[y]

    # Check if the input is a iterable
    elif isinstance(y, Iterable):
        d = [c[i] for i in y]
    else:
        raise KeyError
    return d


def filter_countries(df_val: dict, stone_type=None):
    """
    Returns two dataframe dicts, one with stones whose origin is in the list of countries and one whose origin isn't.
    args:
        df: a dataframe corresponding the the val entrz "FinalOrigin"
        stone_type:
    returns:
        country_df: a val dataframe entry "FinalOrigin"
        not_country_df: a val dataframe entry "FinalOrigin"

    """
    stones_with_countries = df_val.loc[
        df_val["FinalOrigin"].isin(target_list("origin", stone_type))
    ]
    stones_without_countries = df_val.loc[
        ~df_val["FinalOrigin"].isin(target_list("origin", stone_type))
    ]

    return stones_with_countries, stones_without_countries


def filter_countries(df_val: dict, stone_type=None):
    """
    Returns two dataframe dicts, one with stones whose origin is in the list of countries and one whose origin isn't.
    args:
        df: a dataframe corresponding the the val entrz "FinalOrigin"
        stone_type:
    returns:
        country_df: a val dataframe entry "FinalOrigin"
        not_country_df: a val dataframe entry "FinalOrigin"

    """
    stones_with_countries = df_val.loc[
        df_val["FinalOrigin"].isin(target_list("origin", stone_type))
    ]
    stones_without_countries = df_val.loc[
        ~df_val["FinalOrigin"].isin(target_list("origin", stone_type))
    ]

    return stones_with_countries, stones_without_countries


def filter_basaltic_vs_basaltic(df_val: dict, stone_type=None):
    """
    Returns two dataframe dicts, one with stones whose origin is in the list of countries and one whose origin isn't.
    args:
        df: a dataframe corresponding the the val entrz "FinalOrigin"
        stone_type:
    returns:
        country_df: a val dataframe entry "FinalOrigin"
        not_country_df: a val dataframe entry "FinalOrigin"

    """
    # Load xlsx file
    path = hydra.utils.to_absolute_path("data/basaltic burmese.xlsx")
    basaltic_burma_df = pd.read_excel(path, header=None)
    basaltic_burma_df.columns = ["stone_id", "date", "finalised"]
    basaltic_burma_df["stone_id"] = pd.to_numeric(
        basaltic_burma_df["stone_id"], errors="coerce"
    )
    mask = basaltic_burma_df["stone_id"].isna()
    basaltic_burma_df = basaltic_burma_df[~mask]

    basaltic_stones = set(df_val.index) & set(basaltic_burma_df["stone_id"])
    sapphire_basaltic_data = load_sapphires_stones_from_path(
        "data/masterfile_basaltic.h5"
    )
    basaltic_stones = basaltic_stones | set(sapphire_basaltic_data["val"].index)
    df_val.loc[
        df_val["FinalOrigin"].isin(target_list("origin", stone_type)), "FinalOrigin"
    ] = "Big4"
    df_val.loc[df_val.index.isin(basaltic_stones), "FinalOrigin"] = "other"
    other_origins = df_val["FinalOrigin"].isin(["Tanzania", "Cambodia", "Montana"])
    df_val.loc[other_origins, "FinalOrigin"] = "other"

    path = hydra.utils.to_absolute_path("data/sapphire_smaller_origins.xlsx")
    other_origins_df = pd.read_excel(path, header=None)
    other_origins_df.columns = ["stone_id", "date", "finalised", "origin"]
    other_origins_df["stone_id"] = pd.to_numeric(
        other_origins_df["stone_id"], errors="coerce"
    )
    mask = other_origins_df["stone_id"].isna()
    other_origins_df = other_origins_df[~mask]
    df_val.loc[df_val.index.isin(other_origins_df["stone_id"]), "FinalOrigin"] = "other"

    # stones_with_countries = df_val.loc[df_val["FinalOrigin"].isin(target_list("origin", stone_type))]

    # stones_with_countries = df_val.loc[df_val["FinalOrigin"].isin(target_list("origin", stone_type))]
    # stones_without_countries = df_val.loc[~df_val["FinalOrigin"].isin(target_list("origin", stone_type))]
    path = hydra.utils.to_absolute_path("data/basaltic madagascar.xlsx")
    madagascar_df = pd.read_excel(path, header=None)
    madagascar_df.columns = ["stone_id", "date", "origin"]
    madagascar_df["stone_id"] = pd.to_numeric(
        madagascar_df["stone_id"], errors="coerce"
    )
    mask = madagascar_df["stone_id"].isna()
    madagascar_df = madagascar_df[~mask]
    df_val.loc[df_val.index.isin(madagascar_df["stone_id"]), "FinalOrigin"] = "other"

    magascar_index = df_val.loc[df_val["FinalOrigin"] == "Madagascar"].index
    basaltic_magascar_index = df_val.loc[
        df_val.index.isin(madagascar_df["stone_id"])
    ].index
    set(magascar_index) - set(basaltic_magascar_index)

    stones_with_countries = df_val.loc[df_val["FinalOrigin"].isin(["Big4", "other"])]
    stones_without_countries = df_val.loc[
        ~df_val["FinalOrigin"].isin(["Big4", "other"])
    ]

    return stones_with_countries, stones_without_countries


def filter_clarity_enhancement(df, val=True, stone_type="emerald"):
    """
    Returns two dataframes, one with stones whose clarity enhancement is converted to [CE, NCE], and one whose CE isn't.
    """
    # change values from {3a to 3e} to binary {CE, NCE}
    clarity_enhanced_set = {
        "3c high high",
        "1) 3c + C1 2) 3d + C1",
        "3b high",
        "3b high high",
        "3d low low",
        "1) 3d + C1 2) 3c",
        "3d low",
        "3b",
        "3d high",
        "3b to 3d",
        "3c to 3d",
        "3e",
        "1) 3c",
        "3d high high",
        "3b to 3c",
        "3c low low",
        "3b low",
        "3c",
        "3d",
        "3b low low",
        "3c high",
        "3c low",
    }
    not_enhanced_set = {"3a low low", "3a low", "3a1", "3a2"}
    df.loc[
        df["Clarity Enhancement Value"].isin(not_enhanced_set),
        "Clarity Enhancement Value",
    ] = "NCE"
    df.loc[
        df["Clarity Enhancement Value"].isin(clarity_enhanced_set),
        "Clarity Enhancement Value",
    ] = "CE"

    tmp = df.loc[df["Clarity Enhancement Value"].isin(["CE", "NCE"])]
    not_df = df.loc[~df["Clarity Enhancement Value"].isin(["CE", "NCE"])]
    return tmp, not_df


def rename_heat_treatment_columns(df):
    # Replace TE1 with TE
    df.loc[df["Heat Treatment Value"].isin(["TE1"]), "Heat Treatment Value"] = "TE"

    # Replace TE2 with TE
    df.loc[df["Heat Treatment Value"].isin(["TE2"]), "Heat Treatment Value"] = "TE"

    # Replace TE3 with TE
    df.loc[df["Heat Treatment Value"].isin(["TE3"]), "Heat Treatment Value"] = "TE"

    # Replace TE4 with TE
    df.loc[df["Heat Treatment Value"].isin(["TE4"]), "Heat Treatment Value"] = "TE"

    # Replace TE5 with TE
    df.loc[df["Heat Treatment Value"].isin(["TE5"]), "Heat Treatment Value"] = "TE"

    # Replace TE1 high with TE
    df.loc[df["Heat Treatment Value"].isin(["TE1 high"]), "Heat Treatment Value"] = "TE"

    # Replace TE2 high with TE
    df.loc[df["Heat Treatment Value"].isin(["TE2 high"]), "Heat Treatment Value"] = "TE"

    # Replace TE3 high with TE
    df.loc[df["Heat Treatment Value"].isin(["TE3 high"]), "Heat Treatment Value"] = "TE"

    # Replace TE4 high with TE
    df.loc[df["Heat Treatment Value"].isin(["TE4 high"]), "Heat Treatment Value"] = "TE"

    # Replace TE1 low with TE
    df.loc[df["Heat Treatment Value"].isin(["TE1 low"]), "Heat Treatment Value"] = "TE"

    # Replace TE2 low with TE
    df.loc[df["Heat Treatment Value"].isin(["TE2 low"]), "Heat Treatment Value"] = "TE"

    # Replace TE3 low with TE
    df.loc[df["Heat Treatment Value"].isin(["TE3 low"]), "Heat Treatment Value"] = "TE"

    # Replace TE4 low with TE
    df.loc[df["Heat Treatment Value"].isin(["TE4 low"]), "Heat Treatment Value"] = "TE"
    return df


def filter_heat_treatment(df, val=True, stone_type=None):
    """
    Returns two dataframes, one with stones whose heat treatment is [NTE,TE] and one whose heat treatment isn't.
    """
    df = rename_heat_treatment_columns(df)
    tmp = df.loc[df["Heat Treatment Value"].isin(["TE", "NTE"])]
    not_df = df.loc[~df["Heat Treatment Value"].isin(["TE", "NTE"])]
    # df [("Heat Treatment Value","","")] =  df[("Heat Treatment Value","","")] > 0
    # df [("Heat Treatment Value","","")] = df[("Heat Treatment Value","","")].astype('category')
    return tmp, not_df

    # df  = df.loc[df[("Heat Treatment Value","","")].isin([0,1])]
    # not_df = df.loc[~df[("Heat Treatment Value","","")].isin([0,1])]
    # return tmp, not_df


def cfg_to_col(name):
    """
    Map the cfg name to the column name in the dataframe
    """
    cfg_to_col = {"origin": "Origin", "treatment": "Heat Treatment Value"}
    return cfg_to_col[name]


# Return the target tuple in the dataframe given the cfg.target
def target_tuple(name):
    """
    Return the correct entry in the val dataframe given the cfg.target
    """
    if name == "origin" or name == "pre_origin":
        return "FinalOrigin"
    elif name == "treatment":
        return "Heat Treatment Value"
    elif name == "enhancement":
        return "Clarity Enhancement Value"
    else:
        raise ValueError("target name not valid")


def interpolate_ftir_data(ftir_data):
    """Take ftir data as input and output interpolated data at integer wavelength values"""

    x = ftir_data.columns.values
    x_low = math.ceil(x[0])
    x_high = math.ceil(x[-1])

    y = ftir_data.values

    interpolation_func = interp1d(x, y, kind="cubic", assume_sorted=True)

    new_x = np.array(range(x_low, x_high))
    new_y = interpolation_func(new_x)

    assert new_y.shape == (len(ftir_data), len(new_x))

    df = pd.DataFrame(new_y)
    df.columns = new_x
    df.columns.name = "Wavenumber"
    df.index = ftir_data.index

    return df


def get_unique_df(df, source):
    """
    Get proper dataframe for training for a given data source between ED, UV, and ICP.
    FTIR cannot handled here because multiple experiments are performed for the same ID.
    """
    if not source in ["ed", "uv", "icp", "ftir"]:
        raise ValueError("Source must be one of ed, uv, icp to call get_proper_df")
    df = df.copy()
    df = df[(~df.isna().all(axis=1))]
    # df = df[~df.duplicated()]
    df = df.drop_duplicates()
    # df = df[~df.index.duplicated(keep='first')].copy()
    return df


def get_duplicate_IDs_ruby_ftir():
    """
    Return a list of IDs for which FTIR data is not reliable
    """
    with open(
        hydra.utils.to_absolute_path("data/ruby/ruby_ftir_unreliable_Ids.json")
    ) as f:
        json_data = json.load(f)

    return list(set([x for l in json_data for x in l]))


def get_unique_ftir_df(df, stone_type):
    """
    Returns a pandas dataframe containin all IDs for which there exists FTIR data for a given stone type.
    args:
        df: ftir pandas dataframe with all IDs for a given stone type and target
        stone_type: str, one of "sapphire", "ruby"
        raw_path: if not None, load the raw data instead of the processed data (only for ruby)
    returns:
        subset_df: a subset of the input pandas dataframe with all IDs that have FTIR DATA. The dataframe is unique (no duplicates IDs)
    """

    if stone_type == "sapphire" or stone_type == "emerald":
        # In case of sapphires we can apply the same logic as for UV
        df = df.copy()
        df = df[(~df.isna()).any(axis=1)]
        df = df[~df.duplicated()]
        df = df.drop_duplicates()
        df = df[~df.index.duplicated(keep="first")].copy()
        return df

    elif stone_type == "ruby":
        # We have FTIR1_DATA and FTIR2_DATA
        sample_check = df.iloc[:0]
        ftir1_cols = set(sample_check.loc[:, "FTIR1_DATA"].columns)
        ftir2_cols = set(sample_check.loc[:, "FTIR2_DATA"].columns)

        # Assert that FTIR2_DATA_columns contains all FTIR1_DATA_columns
        assert ftir1_cols - ftir2_cols == set()
        df = df.copy()

        df1 = df[(~df.loc[:, "FTIR1_DATA"].isna()).any(axis=1)]
        df1 = df1[~df1.loc[:, "FTIR1_DATA"].duplicated()]
        df1 = df1.drop_duplicates()
        df1 = df1[~df1.index.duplicated(keep="first")].copy()

        df2 = df[(~df.loc[:, "FTIR2_DATA"].isna()).any(axis=1)]
        df2 = df2[~df2.loc[:, "FTIR2_DATA"].duplicated()]
        df2 = df2.drop_duplicates()
        df2 = df2[~df2.index.duplicated(keep="first")].copy()
        total_unique_idx = set(df1.index) | set(df2.index)
        df = df.loc[total_unique_idx]
        df = df[~df.index.duplicated(keep="first")].copy()

        # Get rid of IDs with identical FTIR data
        IDs = get_duplicate_IDs_ruby_ftir()
        df = df.loc[~df.index.isin(IDs)]

        # print("WARNING REMOVE ME!")
        # df = df.loc[[x>19000000 for x in df.index]]
        return df

    else:
        raise NameError("stone_type must be one of sapphire or ruby")


def get_image_dict(df, stone_type, year_subset=None):
    """
    Returns a pandas dataframe containin all IDs for which there exists image data data for a given stone type.
    args:
        df: pandas dataframe with all IDs for a given stone type and target
        stone_type: str, one of "sapphire", "ruby"
        year_subset: list of ints, the years to consider (default: None)
    returns:
        subset_df: a subset of the input pandas dataframe with all IDs that have image data. The dataframe is unique (no duplicates IDs)
    """
    if stone_type == "sapphire":
        image_path = return_sapphire_image_path()
    elif stone_type == "ruby":
        image_path = return_ruby_image_path()
    elif stone_type == "emerald":
        image_path = return_emerald_image_path()
    else:
        raise NameError("stone_type must be one of sapphire or ruby")
    lower_images = glob.glob(
        hydra.utils.to_absolute_path(os.path.join(image_path, "*.jpg")), recursive=True
    )
    upper_images = glob.glob(
        hydra.utils.to_absolute_path(os.path.join(image_path, "*.JPG")), recursive=True
    )
    images = lower_images + upper_images
    first_eight_digits = defaultdict(list)
    for p in images:
        curr = Path(p).stem[:8]
        try:
            first_eight_digits[int(curr)].append(p)
        except ValueError:
            pass

    # Intersect IDs from the pandas dataframe and image IDs
    IDs_with_images = set(df.index) & first_eight_digits.keys()

    # This flag is used to check if we want only to consider a subset of years for images
    if year_subset:
        to_consider = [str(x) for x in range(year_subset % 2000, 20)]
        IDs_with_images = [x for x in IDs_with_images if str(x)[:2] in str(to_consider)]

    # IDs_with_images = {x:v for x,v in first_eight_digits.items() if x in IDs_with_images}
    subset_df = df.loc[IDs_with_images].copy()
    subset_df = subset_df.loc[~subset_df.index.duplicated(keep="first")]
    fin_list = {x: v for x, v in first_eight_digits.items() if x in subset_df.index}
    # fin_list is a dictionary with keys as IDs and values as lists of image paths (i.e. 43545: [path1, path2, ...], 43531: [path1]), converted to a pandas dataframe
    to_list = [(ID, p) for ID in fin_list.keys() for p in fin_list[ID]]
    fin_df = pd.DataFrame(to_list, columns=["ID", "path"])
    fin_df.index = fin_df["ID"]

    # fin_df = fin_df.loc[~fin_df.index.duplicated(keep='first')]
    return fin_df[["path"]]

    subset_df["paths"] = list(fin_list.values())
    return subset_df[["paths"]]  # We want a dataframe, not a series


def lims_source_to_paths_dict(src):
    """Convert the string received from LIMS to a dictionary containing the file paths"""
    match = re.search(r"\d{8}", src)
    # If-statement after search() tests if it succeeded
    if match:
        report_nr = match.group()
    paths_dict = {}
    # remove first and last bracket
    src = src[1:-1]
    # extract infos from {...} within the brackets
    sources = re.findall(r"\{.*?\}", src)
    base_dir = "/run/user/1000/gvfs/smb-share:server=guzlusrv88,share=repository$"
    for source in sources:
        assert source[0] == "{" and source[-1] == "}", "file_paths_parsing_error"
        source = source[1:-1]
        words = source.split(":")
        key = (words[0])[1:-1]
        value = (words[1])[1:-1]
        assert report_nr in value, "filename_report_nr_error"
        value = value.replace("\\", "/")
        value = value.replace("//guzlusrv88/repository$", base_dir)
        if key in paths_dict.keys():
            paths_dict[key] += [value]
        else:
            paths_dict[key] = [value]
    return paths_dict, int(report_nr)


def fill_in_missing_data_sources(stone_data, stone_type):
    empty_df_dict = load_client_data_structure(stone_type)
    for key in empty_df_dict:
        if key not in stone_data:
            stone_data[key] = empty_df_dict[key]
    return stone_data


def return_number_id(s):
    try:
        ID = utils.reg_selector(str(s), r"^\d+")
    except:
        print("Exception while using the reg selector for ICP on row {}".format(s))
        return np.nan
    if len(ID) != 8:
        print("ICP ID from {} does not have 8 digits".format(s))
        return np.nan
    else:
        return int(ID)


def return_df_for_paper(raw_df, target, filter_noisy=False, years=None):
    """
    Args:
        raw_df: from utils.load_client_data()
    """
    if years is None:
        years = ["21", "20", "19", "18"]  #
    # Filter based on the selection of sources
    if target == "origin":
        sources = ["icp", "ed", "uv"]  # Relevant sources for origin
        gt = ["val"]
        sources = sources + gt

        available_ids = raw_df[sources[0]].index.values
        # else:
        #     available_ids = raw_df[sources[0]].index.values
        #     for source in sources:
        #         available_ids = np.union1d(raw_df[source].index.values,available_ids)

        for i in range(1, len(sources)):
            available_ids = np.intersect1d(
                available_ids, raw_df[sources[i]].index.values
            )

        tmp = {}
        for key in sources:
            tmp[key] = raw_df[key].loc[available_ids]

        bool_cond = [x for x in tmp["val"].index if str(x)[:2] in years]

        for key in sources:
            tmp[key] = tmp[key].loc[bool_cond]

        bool_cond = tmp[key]["Origin"].isin(target_list("origin", "sapphire"))
        for key in sources:
            tmp[key] = tmp[key].loc[bool_cond]

        if filter_noisy:
            bool_to_keep = tmp["val"]["Considered_Validation"]
            for key in sources:
                tmp[key] = tmp[key].loc[bool_to_keep]

    elif target == "treatment":
        sources = ["uv", "ftir"]  # "val"]
        gt = ["val"]
        sources = sources + gt

        # if sources:
        available_ids = raw_df[sources[0]].index.values

        for i in range(1, len(sources)):
            available_ids = np.intersect1d(
                available_ids, raw_df[sources[i]].index.values
            )

        tmp = {}
        for key in sources:
            tmp[key] = raw_df[key].loc[available_ids]

        if isinstance(years, list):
            bool_cond = [x for x in tmp["val"].index if str(x)[:2] in years]

            for key in sources:
                tmp[key] = tmp[key].loc[bool_cond]

        bool_cond = tmp[key]["Heat Treatment Value"].isin(
            target_list("treatment", "sapphire")
        )
        for key in sources:
            tmp[key] = tmp[key].loc[bool_cond]

        if filter_noisy:
            bool_to_keep = tmp["val"]["Microscopy & Raman (Treatment)"].isin(
                ["NTE >", "TE >", "TE"]
            )

            for key in sources:
                tmp[key] = tmp[key].loc[bool_to_keep]

            known_issues = [
                18060211,
                19072056,
                20010004,
                20020270,
                19038015,
                20010013,
                21020020,
                21020148,
                20050042,
            ]  # Stones for which the two experts disagree
            for key in sources:
                tmp[key] = tmp[key].loc[~tmp[key].index.isin(known_issues)]
            corrupted = [18062024, 18062038, 19070181]  # FTIR Missing
            for key in sources:
                tmp[key] = tmp[key].loc[~tmp[key].index.isin(corrupted)]

        # Keep only with an origin
        bool_cond = (
            tmp[key]["Origin"] != "not determinable"
        )  # .isin(target_list("origin", "sapphire"))
        for key in sources:
            tmp[key] = tmp[key].loc[bool_cond]

    else:
        raise KeyError("target must be either 'origin' or 'treatment'")
    return tmp


def return_goal_list(row, candidates=None):
    res = []
    if not row:
        return res
    else:
        for target in candidates:
            if target in row:
                res.append(target)
        return res


def return_normalized_heat_treamtn(row):
    if row in ["NTE >", "NTE>", "NTE"]:
        return ["NTE"]
    elif row in ["TE >", "TE>", "TE"]:
        return ["TE"]
    elif row in ["not conclusive", "NTE > TE", "NTE >TE"]:
        return ["not conclusive", ""]
    else:
        return "to get rid of"


def get_subconclusion_matrix(df, filter_ids=None, subconclusion_type="origin"):
    subconclusions = {
        "origin": {
            "icp": "1. Choice LA-ICPMS",
            "ed": "1. Choice Chemical Fingerprint (XRF)",
            "uv": "1. Choice Spectral Fingerprint (Origin)",
            "micro": "1. Choice Microscopy & Raman (Origin)",
        },
        "treatment": {
            "micro": "Microscopy & Raman (Treatment)",
            "ftir": "Spectral Fingerprint (Treatment)",
        },
    }

    if filter_ids is None:
        filter_ids = df.index

    target_column = (
        "Origin" if subconclusion_type == "origin" else "Heat Treatment Value"
    )
    ground_truth = df.loc[filter_ids, target_column]
    target_values = target_list(target=subconclusion_type, stone_type="sapphire")
    results, stats, correct_results = {}, {}, {}

    for sub_conclusion, entry in subconclusions[subconclusion_type].items():
        current_col = df[entry]
        result = (
            current_col.apply(return_goal_list, candidates=target_values)
            if subconclusion_type == "origin"
            else current_col.apply(return_normalized_heat_treamtn).loc[
                current_col != "to get rid of"
            ]
        )
        results[sub_conclusion] = result
        stats[sub_conclusion] = result.apply(len)
        correct_results[sub_conclusion] = pd.Series(
            [
                (1 if val in result.iloc[idx] else 0)
                for idx, val in enumerate(ground_truth)
            ],
            index=current_col.index,
        )

    if subconclusion_type == "origin":
        uv_ed_experts = pd.concat([results["uv"], results["ed"]], axis=1)
        uv_ed_expert_results = []
        for _, row in uv_ed_experts.iterrows():
            uv_entries, ed_entries = set(row.iloc[0]), set(row.iloc[1])
            curr_result = (
                uv_entries & ed_entries
                if (len(uv_entries) == 1) | (len(ed_entries) == 1)
                else {"Kashmir", "Madagascar", "Sri Lanka", "Burma"}
            )
            uv_ed_expert_results.append(list(curr_result))
        uv_ed_experts = pd.Series(uv_ed_expert_results, index=uv_ed_experts.index)
        stats["uv+xrf"] = uv_ed_experts.apply(len)
        correct_results["uv+xrf"] = pd.Series(
            [
                (1 if val in uv_ed_experts.iloc[idx] else 0)
                for idx, val in enumerate(ground_truth)
            ],
            index=ground_truth.index,
        )

    representations = compute_representation_expert_subconclusion(stats)
    accuracy = compute_accuracy_expert_subconclusion(correct_results, stats)
    summary = pd.concat([representations, accuracy], axis=1).sort_index(axis=1)
    single_pred_df = pd.DataFrame(
        {k: v.loc[v == 1] * correct_results[k].loc[v == 1] for k, v in stats.items()}
    )

    # Rename ed_acc and ed_rep to xrf_acc and xrf_rep
    summary = summary.rename(columns={"ed_acc": "xrf_acc", "ed_rep": "xrf_rep"})

    # Rename ftir_acc and ftir_rep to ftir_uv_acc and ftir_uv_rep since in reality the expert used both FTIR and UV
    summary = summary.rename(
        columns={"ftir_acc": "uv+ftir_acc", "ftir_rep": "uv+ftir_rep"}
    )
    return summary, single_pred_df


def compute_representation_expert_subconclusion(statistics):
    reps = {
        key: {k: v / len(value) for k, v in sorted(Counter(value).items())}
        for key, value in statistics.items()
    }
    df_reps = pd.DataFrame(reps)
    df_reps.columns = [x + "_rep" for x in df_reps.columns]
    return df_reps


def compute_accuracy_expert_subconclusion(is_correct_dict, statistics):
    accuracy = {
        key: {
            candidates: sum(is_correct_dict[key].loc[ids]) / len(ids)
            for candidates, ids in statistics[key]
            .groupby(statistics[key])
            .groups.items()
        }
        for key in is_correct_dict.keys()
    }
    df_accuracy = pd.DataFrame(accuracy)
    df_accuracy.columns = [x + "_acc" for x in df_accuracy.columns]
    return df_accuracy


def load_results(results, available_ids):

    try:
        df_test = results.return_subsect_roll(0, mode="test")
    except AttributeError:
        df_test = results.return_subsect_roll(0)

    df_test["confidence"] = df_test["Pred"].max(axis=1)
    df_test = df_test.loc[df_test.index.get_level_values(1).isin(available_ids)]
    try:
        df_train = results.return_subsect_roll(0, mode="train")
        df_train["confidence"] = df_train["Pred"].max(axis=1)
    except AttributeError:
        df_train = None

    try:
        df_val = results.return_subsect_roll(0, mode="val")
        df_val = df_val.loc[df_val.index.get_level_values(1).isin(available_ids)]
        df_val["confidence"] = df_val["Pred"].max(axis=1)
    except AttributeError:
        df_val = df_test

    return df_train, df_val, df_test


def return_threshold(results, accuracy=0.95, available_ids=None):
    df_train, df_val, df_test = load_results(results, available_ids=available_ids)
    df_train = df_train.loc[df_train.index.get_level_values(1).isin(available_ids)]
    # Concatenate train and val for thresholding
    df = pd.concat([df_train, df_val], axis=0)
    # df = df_val
    # df = df_val
    df["confidence"] = df["Pred"].max(axis=1)
    df = df.sort_values("confidence", ascending=False)

    thresholds = []
    for cv in df.index.levels[0]:
        xxx = df.loc[cv]
        for i in reversed(range(len(xxx))):
            value = results.compute_accuracy_score(xxx.iloc[:i], exclude_false_val=True)

            try:
                confidence_value = xxx.iloc[:i]["confidence"].iloc[-1]
            except:
                confidence_value = 1.0

            if value > accuracy:  # target_accuracy*1.03:
                break

        thresholds.append(confidence_value)

    return thresholds


# Helper functions
def filter_data_for_task(raw_data, task, years=None):
    filtered_data = return_df_for_paper(raw_data, task, filter_noisy=True, years=years)
    available_ids = filtered_data["val"].index
    return filtered_data, available_ids


def get_subconclusion(filtered_data, available_ids, task):
    subconclusion, one_pred = get_subconclusion_matrix(
        filtered_data["val"], filter_ids=available_ids, subconclusion_type=task
    )
    return subconclusion, one_pred


def remove_disagreeing_stones(one_pred, available_ids):
    disagreeing_indices = one_pred.loc[
        (one_pred["icp"] == False) | (one_pred["micro"] == False)
    ].index
    return [x for x in available_ids if x not in disagreeing_indices]


def to_numeric_icp(df_icp):
    tmp = df_icp.apply(pd.to_numeric, errors="coerce")
    for col in tmp.columns:
        tmp = tmp.loc[~(tmp[col] < 0)]
    tmp = tmp.loc[~(tmp["Al27"] < 500000)]
    tmp = tmp.loc[~(tmp["Nd146"] > 10)]
    df_icp = tmp
    return df_icp


def extract_date(id):
    """
    Given the id, extract the date (i.e. first two digits are the year and the next two are the month)
    """
    yy = str(id)[:2]
    mm = str(id)[2:4]
    # Convert to date
    import datetime

    date = datetime.date(int("20" + yy), int(mm), 1)
    return date
