"""
Code auto formatted with autoformatter named black.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

def filter_path_for_target_label(paths_txt, paths_npy, target_label):
    paths_txt = [
        label_path for label_path in paths_txt if f"__{target_label}" in label_path
    ]
    paths_npy = [
        label_path for label_path in paths_npy if f"__{target_label}" in label_path
    ]
    return paths_txt, paths_npy



def load_labels_and_data_from_npy(file_paths_npy_all):
    """
    This function is designed for loading labels and data given a filepath

    Parameters
    --
    file_paths_npy_all [list]: of the corresponding song .npy ; this will do a replace .npy by .txt so txt file must have the same name but not the format

    returns: list of labels  of all songs and list of data npy formatted (which is what you laoaded)
    """
    labels_list, data_npy_list = list(), list()
    for file_path_npy in file_paths_npy_all:
        file_path_txt = file_path_npy.replace(".npy", ".txt")  
        assert os.path.exists(
            file_path_npy
        ), f"[ERROR] The txt file with path {file_paths_npy} doesnt exist"
        assert os.path.exists(
            file_path_txt
        ), f"[ERROR] The txt file with path {file_paths_txt} doesnt exist"

        data = np.load(file_path_npy)
        with open(file_path_txt, "r") as f:
            labels = [x.strip().replace("\n", "") for x in f.readlines()]
        data_npy_list.append(data)
        labels_list.append(labels)
        print(
            f"User must verify if the following files correspoond eaco other :\n {file_path_txt} \n {file_path_npy}"
        )
    return labels_list, data_npy_list



def merge_numpy_data(data_npy_list: list):
    """
    Merge a list of numpy files named: data_npy_list into a bigger numpy file
    Parameter:
    data_npy_list [list]: list of npy files; with equal shape
    Returns:
    big_numpy -> merged numpy files from list data_npy_list
    """
    big_numpy = data_npy_list[0]
    for idx in range(1, len(data_npy_list), 1):
        big_numpy = np.append(big_numpy, data_npy_list[idx], axis=0)
    return big_numpy


def merge_labels_data(labels_list: list):
    """
    Function for merging labels in a bigger list
    """
    big_labels_list = list()
    for lab in labels_list:
        big_labels_list.extend(lab)
    return big_labels_list


def trainvaltest_split(
    X, y, train_ratio: float, validation_ratio: float, test_ratio: float, random_state
):
    """
    Split the data set into train val test
    --
    Params
    X [pd.DataFrame]; features to split
    y [pd.Series] ; target to split
    train_ratio [float]; fractio of train
    validation_ratio [float]; fractio of val
    test_ratio [float]; fractio of test
    random_state [int]; random_seed for splitting
    ---
    Returns features for trainvaltest; targets for trainvaltest respectively:
    x_train, x_val, x_test,y_train,y_val,y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=random_state
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test