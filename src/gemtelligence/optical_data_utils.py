import pandas as pd
from .utils import reg_selector, reg_filter
import pathlib
from collections import Counter
from typing import List, Union, Dict
import pdb


def collect_optical_spectroscopy_based_on_regex(
    path: str, regex=None
) -> List[Union[str, pathlib.Path]]:
    res = list()
    for name_origin in pathlib.Path(path).glob("*"):
        dest = list(pathlib.Path(name_origin).glob("*"))
        if regex:
            dest = reg_filter(dest, regex)
        des = [(x, name_origin.stem) for x in dest]
        res.extend(des)
    return res


def check_first_column(df):
    if type(df) == pathlib.PosixPath:
        res = pd.read_csv(df.resolve())
    elif type(df) == pd.DataFrame:
        res = df
    elif type(df) == str:
        res = pd.read_csv(df)
    else:
        raise TypeError("Type not supported", type(df))
    begin = float(res.at[1, res.columns[0]])
    try:
        end = float(res.at[1201, res.columns[0]])
    except:
        end = float("nan")
    return begin, end


def check_start_and_end(files):
    res = Counter()
    for l, _ in files:
        begin, end = check_first_column(l)
        tmp = "_".join([str(begin), str(end)])
        res.update([tmp])
    return res


def return_path_if_end_is(files, s):
    res = []
    for l, _ in files:
        begin, end = check_first_column(l)
        if str(end) == s:
            res.append(l)
    return res


def count_number_of_columns(files):
    res = Counter()
    for l, _ in files:
        t = count_number_columns(l)
        res.update(str(t))
    return res


def count_number_columns(path):
    df = pd.read_csv(path.resolve())
    res = len(df.columns)
    return res


def return_path_if_col_num(files, col=0):
    res = []
    for l, _ in files:
        t = count_number_columns(l)
        if col == t:
            res.append(l)
    return res
