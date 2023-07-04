from neural_networks import utility_functions as ut
import torch
import pandas as pd
import xgboost as xgb
from plots import *



def get_data_variables(xarray_data):
    """
    Extracts data variables from an xarray dataset and processes them. If the data variable has more than one dimension,
    the mean is taken over all dimensions. Further NaN values are replaced with non-NaN values from the other dimensions.

    Args:
        xarray_data (xarray.Dataset): Input xarray dataset.

    Returns:
        dict: A dictionary containing processed data variables as numpy arrays.

    """

    # Get the data variable names
    data_variables = xarray_data.data_vars.keys()

    # Dictionary to store processed data variables
    variables_data = {}

    # Iterate over each data variable
    for var_name in data_variables:
        # Convert data variable to a torch tensor
        var_data = torch.tensor(xarray_data[var_name].values)

        # Check the dimensions of the variable data
        if len(var_data.shape) > 1:
            # Check for NaN values
            nan_mask = torch.isnan(var_data)
            nan_indices = torch.where(nan_mask)
            non_nan_indices = torch.where(~nan_mask)

            # Replace NaN values with non-NaN values from the other dimensions
            for dim_idx in range(1, len(var_data.shape)):
                var_data[nan_indices[0], nan_indices[dim_idx]] = var_data[non_nan_indices[0], non_nan_indices[dim_idx]][0]

            # Compute the mean over all dimensions
            var_data = torch.mean(var_data, dim=0)

        # Convert the processed data variable back to a numpy array
        variables_data[var_name] = var_data.numpy()

    return variables_data

def normalize_data(data):
    """
    Normalize the data using mean and standard deviation.

    Args:
        data (torch.Tensor): Input data to be normalized.

    Returns:
        torch.Tensor: Normalized data.

    """

    # Calculate the mean and standard deviation along the feature dimension
    mean = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)

    # Apply normalization using the mean and standard deviation
    normalized_data = (data - mean) / std

    return normalized_data


def dictionary_to_dataframe(data_dict):
    """
    Convert a dictionary to a pandas DataFrame.

    Args:
        data_dict (dict): Input dictionary with category keys and numpy arrays as values.

    Returns:
        pd.DataFrame: A pandas DataFrame where each entry in the numpy arrays becomes a row, and categories are columns.

    """

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over the dictionary items
    for category, values in data_dict.items():
        # Add a new column to the DataFrame with category as the column name
        df[category] = values

    return df

def remove_nan_columns(input_data, target_data):
    """
    Remove columns with NaN values from the input data

    Args:
        input_data (pd.DataFrame): Input data with features and target variable.

    Returns:
        pd.DataFrame: Input data without NaN columns.

    """

    # Forward fill NaN values in the target data and input data
    input_data = input_data.ffill()
    target_data = target_data.ffill()

    # Identify columns with NaN values in the input data
    nan_cols = show_nan(input_data)

    # Drop columns with NaN values from the input data
    for col_name in nan_cols:
        input_data = input_data.drop(columns=[col_name])

    # Return the input data and target data without the NaN columns
    return input_data, target_data

def show_nan(df: pd.DataFrame) -> list:
    """ Create a dictionary to store the indexes of nan values.

        Args:
            df: the data frame where to look for nans.
        Returns:
            A dictionary with nan/null indexes for every column.

    """
    results = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            results.append(col)

    return results


def get_top_features_pearson(data, target, num_features):
    """
    Get the top features based on Pearson's correlation coefficient with the target variable.

    Args:
        data (pd.DataFrame): Input data with features.
        target (pd.Series): Target variable.
        num_features (int): Number of top features to return.

    Returns:
        list: A list of the top 'num_features' features based on Pearson's correlation coefficient.

    """

    print(data)

    # Calculate the Pearson's correlation coefficient for each feature
    correlations = data.corrwith(target)

    # Sort the correlations in descending order
    sorted_correlations = correlations.abs().sort_values(ascending=False)

    # Get the top 'num_features' features
    top_features = sorted_correlations.index[:num_features].tolist()

    return top_features

def df_shifted(df, target=None, lag=0):
    """
    Shifts the values of the columns in a DataFrame by a specified lag period.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Name of the target variable column to be preserved without shifting.
        lag (int): Number of periods to shift the values by.

    Returns:
        pd.DataFrame: A new DataFrame with shifted values.

    """

    # If no lag and no target variable specified, return the original DataFrame
    if not lag and not target:
        return df

    # Create a dictionary to store the shifted columns
    new = {}

    # Iterate over the columns in the DataFrame
    for column in df.columns:
        if column == target:
            # Preserve the target variable without shifting
            new[column] = df[target]
        else:
            # Shift the values of the column by the specified lag period
            new[column] = df[column].shift(periods=lag)

    # Create a new DataFrame using the shifted values
    return pd.DataFrame(data=new)


def get_top_features_xgboost(input_data, target_data, num_features):
    """
    Get the top features based on feature importance scores from an XGBoost model.

    Args:
        input_data (pd.DataFrame): Input data with features.
        target_data (pd.Series): Target variable.
        num_features (int): Number of top features to return.

    Returns:
        list: A list of the top 'num_features' features based on feature importance scores.

    """

    # Create an XGBoost regression model
    model = xgb.XGBRegressor()
    model.fit(input_data, target_data)

    # Get feature importance scores from the model
    feature_importance = model.feature_importances_

    # Create a DataFrame to store feature importance information
    importance_df = pd.DataFrame({'Feature': input_data.columns, 'Importance': feature_importance})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Get the top 'num_features' features
    top_features = importance_df.head(num_features)['Feature'].tolist()

    return top_features

def get_top_features_spearman(data, target, num_features):
    """
    Get the top features based on Spearman's rank correlation coefficient with the target variable.

    Args:
        data (pd.DataFrame): Input data with features.
        target (pd.Series): Target variable.
        num_features (int): Number of top features to return.

    Returns:
        list: A list of the top 'num_features' features based on Spearman's rank correlation coefficient.

    """

    # Calculate the Spearman's rank correlation coefficient for each feature
    correlations = data.corrwith(target, method='spearman')

    # Sort the correlations in descending order
    sorted_correlations = correlations.abs().sort_values(ascending=False)

    # Get the top 'num_features' features
    top_features = sorted_correlations.index[:num_features].tolist()

    return top_features


def get_top_features_time_lag_corr(data, target_variable, time_lag, num_features):
    """
    Get the top features based on time-lagged correlation with the target variable.

    Args:
        data (pd.DataFrame): Input data with features.
        target_variable (str): Name of the target variable.
        time_lag (int): Number of time periods to lag the data.
        num_features (int): Number of top features to return.

    Returns:
        list: A list of the top 'num_features' features based on time-lagged correlation.

    """
    if time_lag != 0:
        # Shift the data by the specified time lag
        data = df_shifted(data, target=target_variable, lag=time_lag)

    # Calculate the correlation with the target variable
    correlations = data.corr()[target_variable]

    # Sort the correlations in descending order
    sorted_correlations = correlations.sort_values(ascending=False)

    # Get the top 'num_features' features
    top_features = sorted_correlations[:num_features].index.tolist()

    return top_features


def feature_selection_func(data, target, target_variable, feature_selection_type, num_features, time_lag):
    """
    Perform feature selection based on the specified method.

    Args:
        data (DataFrame): Input data.
        target (Series): Target variable.
        target_variable (str): Name of the target variable.
        feature_selection_type (str): Type of feature selection method.
        num_features (int): Number of top features to select.
        time_lag (int): Time lag for time lag correlation method.

    Returns:
        list: List of top selected features.
    """

    top_features = None

    if feature_selection_type == "pearson":
        top_features = get_top_features_pearson(data, target, num_features)
        print("Top features based on Pearson correlation:\n{}".format(top_features))

    elif feature_selection_type == "xgboost":
        top_features = get_top_features_xgboost(data, target, num_features)
        print("Top features based on XGBoost feature importance:\n{}".format(top_features))

    elif feature_selection_type == "spearman":
        top_features = get_top_features_spearman(data, target, num_features)
        print("Top features based on Spearman correlation:\n{}".format(top_features))

    elif feature_selection_type == "time_lag_corr":
        top_features = get_top_features_time_lag_corr(data, target_variable, time_lag, num_features)
        print("Top features based on time lag correlation:\n{}".format(top_features))

    else:
        print("Invalid feature selection type. Please choose between 'pearson', 'xgboost', 'time_lag_corr', and 'spearman'.")

    return top_features