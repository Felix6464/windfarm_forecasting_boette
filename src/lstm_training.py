from LSTM_enc_dec import *
from utility_functions import *
from data_preprocessing import normalize_data

# Load data to be used for training
data = pd.read_csv("./preprocessed_data/filtered_dataset_brazil_time_lag_corr.csv")
data = np.array(data).T
data = torch.from_numpy(data)

# Calculate the mean and standard deviation along the feature dimension
data = normalize_data(data)


# Splitting data into training and testing
index_train = int(0.8 * len(data[0, :]))
data_train = data[:, :index_train]
data_test = data[:, index_train:]

# Setting input and output window sizes for data
input_window = 6
output_window = 1
num_features = 11
stride = 1

input_data, target_data = dataloader_seq2seq_feat(data_train,
                                                  input_window=input_window,
                                                  output_window=output_window,
                                                  stride=stride,
                                                  num_features=num_features)

input_data_test, target_data_test = dataloader_seq2seq_feat(data_test,
                                                            input_window=input_window,
                                                            output_window=output_window,
                                                            stride=stride,
                                                            num_features=num_features)

# convert windowed data from np.array to PyTorch tensor
data_train, target, data_test, target_test = numpy_to_torch(input_data, target_data, input_data_test, target_data_test)


# Setting hyperparameters for training
hidden_size = 256
num_layers = 3
learning_rate = 0.001
num_epochs = 100
input_window = 6
output_window = 1
batch_size = 64
training_prediction = "mixed_teacher_forcing"
teacher_forcing_ratio = 0
dynamic_tf = True
shuffle = True
loss_type = "L1"
wind_farm = "britain_time_lag_corr_"

print("Start training")

# Specify the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM_Sequence_Prediction(input_size = data_train.shape[2], hidden_size = hidden_size, num_layers=num_layers)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss, loss_test = model.train_model(data_train,
                                    target,
                                    data_test,
                                    target_test,
                                    num_epochs,
                                    input_window,
                                    output_window,
                                    batch_size,
                                    training_prediction,
                                    teacher_forcing_ratio,
                                    learning_rate,
                                    dynamic_tf,
                                    loss_type,
                                    optimizer)


rand_identifier = np.random.randint(0, 10000000)
print(f"Model saved as model_{rand_identifier}.pt")

# Save the model and hyperparameters to a file
parameters = {
        'hidden_size': hidden_size,
        "num_layers": num_layers,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        "input_window": input_window,
        "output_window": output_window,
        "batch_size": batch_size,
        "training_prediction": training_prediction,
        "teacher_forcing_ratio": teacher_forcing_ratio,
        "dynamic_tf": dynamic_tf,
        "loss": loss.tolist(),
        "loss_test": loss_test.tolist(),
        "loss_type": loss_type,
        "shuffle": shuffle,
        "wind_farm": wind_farm,
    }

torch.save({'hyperparameters': parameters,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f'./trained_models/model_{rand_identifier}.pt')


