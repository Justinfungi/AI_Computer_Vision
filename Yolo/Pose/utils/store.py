import numpy as np
import pandas as pd

"""class DataStore():
    x = None
    y = None
    data = DataStore()"""
cols = []
for i in range(1,18):
    for j in ["x","y"]:
        cols.append((i,j))
col_list = pd.MultiIndex.from_tuples(cols)

def create_kpt_df():
    cols = []
    for i in range(1,18):
        for j in ["x","y"]:
            cols.append((i,j))
    col_list = pd.MultiIndex.from_tuples(cols)
    kpt_df = pd.DataFrame(columns=col_list)
    return kpt_df


def modify_kpt_df(kpt_df,kpts):
    lst=[]
    for kid in range(17):
        x_coord, y_coord = kpts[3 * kid], kpts[3 * kid + 1]
        lst.append(0)
        lst.append(0)
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            conf = kpts[3 * kid + 2]
            if conf < 0.5:
                continue
            else:
                lst[-2]=int(x_coord)
                lst[-1]=int(y_coord)
    new = pd.DataFrame(np.array(lst).reshape(1,34),columns=col_list)
    kpt_df = pd.concat((kpt_df,new))
    return kpt_df


def store_kpt_df(kpt_df,source):
    kpt_df.to_excel(f"data/excel/kp_{source[13:]}_output.xlsx")
    return 0
