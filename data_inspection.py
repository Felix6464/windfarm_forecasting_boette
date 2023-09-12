import pandas as pd
from fuzzywuzzy import fuzz
import xarray as xr
import src.data_preprocessing as dp


# Set the windfarm type to "brazil" or "britan"
windfarm_type= "brazil"


if windfarm_type == "britain":
    # Read the CSV file
    data = pd.read_csv('src/raw_data/Turbine_Data_Kelmarsh_2_2020-01-01_-_2021-01-01_229.csv')

else:
    # Load the input data from the NetCDF file
    data = xr.open_dataset('src/raw_data/UEPS_v1.nc')

    # Get the list of variables in the input data
    variable_list = dp.get_data_variables(data)
    print("Variables in the input data:\n{}".format(variable_list))
    # Convert the dictionary of variables to a dataframe
    data = dp.dictionary_to_dataframe(variable_list)
    print("Input Data (Brazil Windfarm):\n{}".format(data))

# Store column names in a list
column_names = data.columns.tolist()

print("There are {} columns in the dataset. \n".format(len(column_names)))
# Iterate over each column
for column in column_names:
    # Access the column data
    column_data = data[column]

    # Print column name
    print("Column Name:", column)

    # Print number of non-null values
    print("Number of Non-Null Values:", column_data.count())

    # Print data type of the column
    print("Data Type:", column_data.dtype)

    try:
        # Print mean value
        print("Mean:", column_data.mean())

        # Print standard deviation
        print("Standard Deviation:", column_data.std())

        # Print minimum value
        print("Minimum:", column_data.min())

        # Print maximum value
        print("Maximum:", column_data.max())

    except TypeError:
        print("Data type not compatible for calculation.")

    # Print some example values
    print("Example Values:")
    for value in column_data.sample(n=5):  # Adjust the number of samples as needed
        print(value)

    print("\n")  # Print a new line for better readability

# Terms for fuzzy matching
terms = ['wind speed', 'wind_direction', 'temperature', 'humidity', 'pressure', 'wind power', 'power']

# Initialize an empty list to store matching column names
matching_columns = []

# Iterate over each column
for column in data.columns:
    # Check fuzzy match with each term
    for term in terms:
        if fuzz.partial_ratio(term, column.lower()) >= 98:
            matching_columns.append(column)
            break  # Skip checking remaining terms if a match is found

# Print the matching column names
print("Matching Column Names for the following terms : {} + {}:".format(terms, matching_columns))