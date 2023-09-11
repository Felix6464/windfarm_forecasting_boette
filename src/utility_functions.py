import pandas as pd
import json
import torch
import numpy as np
import os

from models.LSTM_enc_dec_input import *
from models.LSTM_enc_dec import *
from models.LSTM import *
from models.GRU_enc_dec import *



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

def load_models_testing(num_lstm_base, num_lstm, num_lstm_input, num_gru, num_lstm_input_tf):
    # Load the saved models
    saved_model_lstm_base = torch.load(f"./final_models/model_{num_lstm_base}.pt")
    saved_model_lstm = torch.load(f"./final_models/model_{num_lstm}.pt")
    saved_model_lstm_input = torch.load(f"./final_models/model_{num_lstm_input}.pt")
    saved_model_lstm_input_tf = torch.load(f"./final_models/model_{num_lstm_input_tf}.pt")
    saved_model_gru = torch.load(f"./final_models/model_{num_gru}.pt")

    # Load the hyperparameters of the lstm_model base
    params_lb = saved_model_lstm_base["hyperparameters"]

    # Load the hyperparameters of the lstm_model_enc_dec
    params_l = saved_model_lstm["hyperparameters"]

    # Load the hyperparameters of the lstm_input_model_enc_dec
    params_li = saved_model_lstm_input["hyperparameters"]

    # Load the hyperparameters of the lstm_input_model_enc_dec with teacher_forcing
    params_li_tf = saved_model_lstm_input_tf["hyperparameters"]

    # Load the hyperparameters of the fnn_model
    params_g = saved_model_gru["hyperparameters"]

    # Specify the device to be used for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_lstm_base = LSTM_Sequence_Prediction_Base(input_size=params_lb["num_features"],
                                                    hidden_size=params_lb["hidden_size"],
                                                    num_layers=params_lb["num_layers"])

    model_lstm = LSTM_Sequence_Prediction(input_size=params_lb["num_features"],
                                          hidden_size=params_l["hidden_size"],
                                          num_layers=params_l["num_layers"])

    model_lstm_inp = LSTM_Sequence_Prediction_Input(input_size=params_lb["num_features"],
                                                    hidden_size=params_li["hidden_size"],
                                                    num_layers=params_li["num_layers"])

    model_lstm_inp_tf = LSTM_Sequence_Prediction_Input(input_size=params_lb["num_features"],
                                                       hidden_size=params_li_tf["hidden_size"],
                                                       num_layers=params_li_tf["num_layers"])

    model_gru = GRU_Sequence_Prediction(input_size=params_lb["num_features"],
                                        hidden_size=params_g["hidden_size"],
                                        num_layers=params_g["num_layers"])

    # Load the saved models
    model_gru.load_state_dict(saved_model_gru["model_state_dict"])
    model_gru = model_gru.to(device)
    model_lstm_base.load_state_dict(saved_model_lstm_base["model_state_dict"])
    model_lstm_base = model_lstm_base.to(device)
    model_lstm.load_state_dict(saved_model_lstm["model_state_dict"])
    model_lstm = model_lstm.to(device)
    model_lstm_inp.load_state_dict(saved_model_lstm_input["model_state_dict"])
    model_lstm_inp = model_lstm_inp.to(device)
    model_lstm_inp_tf.load_state_dict(saved_model_lstm_input["model_state_dict"])
    model_lstm_inp_tf = model_lstm_inp.to(device)

    return model_lstm_base, model_lstm, model_lstm_inp, model_gru, model_lstm_inp_tf