# Windfarm Prediction


## Description
This repository contains Python code for processing and analyzing wind farm data using machine learning techniques. The provided scripts and notebooks facilitate data preprocessing, feature selection, and LSTM-based time series prediction.

## Download Raw Data

### British Wind Farm Data
Go to https://zenodo.org/record/5841834#.ZEajKXbP2BQ

Download Kelmarsh_SCADA_2016_3082.zip. (Any other year is fine too) 

Unzip the data and copy the corresponding turbine data into the folder /src/raw_data 

### Brazilian Wind Farm Data

Go to https://zenodo.org/record/1475197#.ZD6iMxXP2WC

Download UEPS_v1.nc

Copy the file into the folder /src/raw_data

## How to run
First, install dependencies
```bash
# clone project   
git clone https://github.com/Felix6464/windfarm_forecasting_boette.git

# install project   
cd windfarm_forecasting 
pip install -e .   
pip install -r requirements.txt

 ```   
Next, navigate to any file and run it.
### `windfarm_britain_dataprocessing.ipynb`

- Reads raw data from the British wind farm dataset into a pandas dataframe.
- Imports the `datapreprocessing.py` file with data processing functions.
- Splits the data into target and input datasets.
- Performs min-max normalization and removes NaN values.
- Allows feature selection using various mechanisms.
- Saves a new CSV file with the data of the selected important features to ./preprocessed_data

### `windfarm_brazil_dataprocessing.ipynb`

- Processes wind farm data from an nc file.
- Similar data processing steps as in `windfarmbritaindataprocessing.ipynb`.

### `lstm_training.py`

- Trains an LSTM model for time series prediction.
- Reads the preprocessed CSV file and performs data normalization.
- Splits the data into train and test sets with specified windows.
- Shapes the data into time series splits for model training.
- Sets hyperparameters and initializes the model.
- Uses teacher-forcing during training to learn temporal dependencies.
- Saves model-related information for later evaluation and plotting.
- in order to change the model that is being used you have to change the respective import of the model
- and specify the class of model initalization in the main function (line 122)

### `LSTM_enc_dec.py`

- Contains the neural network architecture and training loop for LSTM models.
- Includes a function for preparing input data compatible with time series prediction.

### `utility_functions.py`

- Contains two minor functions:
    - Loading text as JSON.
    - Appending two pandas dataframes.

### `plot_saved_model.py`

- Loads a saved model and preprocessed data from files.
- Plots the time series of the first variable (target variable) for training and test data.
- Creates two plots showing the convergence of training and test loss over the specified range of epochs.
- Imports `plots.py`, which contains individual functions for the plots.

### `lstm_testing.py`

- Loads a saved model and a preprocessed dataset (not the dataset the model was trained on).
- Evaluates the trained model on an unseen dataset.

### `global_model_approach.ipynb`

- Appends two preprocessed datasets to create a larger dataset.
- Increases variability in the data and potentially improves performance on unseen datasets.