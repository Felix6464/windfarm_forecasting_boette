import pandas as pd
import json
import torch
import numpy as np
import os



def load_text_as_json(file_path):
    """
    Load text from a file and parse it as JSON.

    Args:
        file_path (str): Path to the text file.

    Returns:
        dict: Parsed JSON data.
    """

    with open(file_path, 'r') as file:
        text = file.read()
        json_data = json.loads(text)

    return json_data


def append_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Append two DataFrames with the same columns to create a new DataFrame.

    Args:
        df1: The first DataFrame.
        df2: The second DataFrame.

    Returns:
        A new DataFrame with the appended data from both input DataFrames.

    """
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    : param Xtrain:                    windowed training input data (input window size, # examples, # features)
    : param Ytrain:                    windowed training target data (output window size, # examples, # features)
    : param Xtest:                     windowed test input data (input window size, # examples, # features)
    : param Ytest:                     windowed test target data (output window size, # examples, # features)
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

    '''

    X_train = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train, Y_train, X_test, Y_test




def save_dict(file_path, dictionary):
    if os.path.exists(file_path):
        # Load existing dictionary from file
        with open(file_path, 'r') as file:
            existing_dict = json.load(file)

        # Merge dictionaries
        merged_dict = dict_merge([existing_dict, dictionary])
        data_to_write = merged_dict
    else:
        # Create a new dictionary with the given dictionary
        data_to_write = dictionary

    # Save dictionary to file
    with open(file_path, 'w') as file:
        json.dump(data_to_write, file)


def dict_merge(dicts_list):
    d = {**dicts_list[0]}
    for entry in dicts_list[1:]:
        for k, v in entry.items():
            d[k] = ([d[k], v] if k in d and type(d[k]) != list
                    else [*d[k], v] if k in d
            else v)
    return d