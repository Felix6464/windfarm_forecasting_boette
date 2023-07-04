from LSTM_enc_dec import *
from data_preprocessing import *

# Specify the model number of the model to be tested
model_num = "1tb"
saved_model = torch.load(f"./trained_models/britain/model_{model_num}.pt")

# Load the hyperparameters of the model
params = saved_model["hyperparameters"]

hidden_size = params["hidden_size"]
num_layers = params["num_layers"]
learning_rate = params["learning_rate"]
input_window = params["input_window"]
output_window = params["output_window"]
batch_size = params["batch_size"]
loss_type = params["loss_type"]


# Load the test data
data = pd.read_csv('./preprocessed_data/filtered_dataset_britain_eval_time_lag_corr.csv')
data = np.array(data).T
data = torch.from_numpy(data)

# Calculate the mean and standard deviation along the feature dimension
data = normalize_data(data)

# Specify the number of features and the stride for generating timeseries data
num_features = 11
stride = 1

input_data_test, target_data_test = dataloader_seq2seq_feat(data,
                                                            input_window=input_window,
                                                            output_window=output_window,
                                                            stride=stride,
                                                            num_features=num_features)


# Specify the device to be used for testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert windowed data from np.array to PyTorch tensor
X_test = torch.from_numpy(input_data_test)
Y_test = torch.from_numpy(target_data_test)

# Initialize the model and load the saved state dict
model = LSTM_Sequence_Prediction(input_size = X_test.shape[2], hidden_size = hidden_size, num_layers=num_layers)
model.load_state_dict(saved_model["model_state_dict"])
model.to(device)

# Evaluate the model one time over whole test data
loss_test = model.evaluate_model(X_test, Y_test, input_window, batch_size, loss_type)
print(f"Test loss: {loss_test}")


