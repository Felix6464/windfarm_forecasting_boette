from models.LSTM_enc_dec import *
from utility_functions import *
from data_preprocessing import *


# Load data to be used for training
data = pd.read_csv("preprocessed_data/filtered_dataset_brazil2_time_lag_corr_144.csv")
data = np.array(data).T
data = torch.from_numpy(data)

# Calculate the mean and standard deviation along the feature dimension
data = normalize_data(data)

num_features = 11

model_num = "8543434np"


saved_model = torch.load(f"./final_models/model_{model_num}.pt")

params = saved_model["hyperparameters"]
hidden_size = params["hidden_size"]
num_layers = params["num_layers"]
input_window = params["input_window"]
output_window = params["output_window"]
loss_type = params["loss_type"]
loss = params["loss_train"]
loss_test = params["loss_test"]


index_train = int(0.8 * len(data[0, :]))
data_train = data[:, :index_train]
data_test = data[:, index_train:]

input_data, target_data = dataloader_seq2seq_feat(data_train,
                                                  input_window=input_window,
                                                  output_window=output_window,
                                                  num_features=num_features)
input_data_test, target_data_test = dataloader_seq2seq_feat(data_test,
                                                            input_window=input_window,
                                                            output_window=output_window,
                                                            num_features=num_features)
print("Data loaded")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert windowed raw_data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = numpy_to_torch(input_data, target_data, input_data_test, target_data_test)
model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)

model.load_state_dict(saved_model["model_state_dict"])
model.to(device)


plot_model_forecast_PC(model, X_train, Y_train, X_test, Y_test, model_num)

print("Hyperparameters Model : {}".format(params))
plot_loss(loss, loss_test, model_num, "Train_Eval_Loss")
