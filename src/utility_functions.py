import pandas as pd
import json



def load_text_as_json(file_path):
    """
    Load text from a file and parse it as JSON.

    Args:
        file_path (str): Path to the text file.

    Returns:
        dict: Parsed JSON data.
    """
    json_data = None

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