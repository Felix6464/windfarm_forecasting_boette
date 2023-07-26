from src.models.LSTM_enc_dec_old import *
from src.plots.plots import *
from utility_functions import *
import torch.utils.data as datat
from torch.utils.data import DataLoader
from data_preprocessing import normalize_data


def main():

    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")
    print("Data shape : {}".format(data.shape))

    # Calculate the mean and standard deviation along the feature dimension
    data = normalize_data(data)
    data = data[:, :30000]

    index_train = int(0.9 * len(data[0, :]))
    data = data[:, index_train:]

    model_num = [("3561908np", "model")]

    id = ["horizon_eval_test"]

    loss_list = []
    loss_list_eval = []
    horizon = True

    for m in range(len(model_num)):
        saved_model = torch.load(f"./trained_models/lstm/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]
        print("Hyperparameters of model {} : {}".format(model_num[m][0], params))
        wandb.init(project=f"ML-Climate-SST-{'Horizon'}", config=params, name=params['name'])

        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        input_window = params["input_window"]
        batch_size = params["batch_size"]
        loss_type = params["loss_type"]
        shuffle = params["shuffle"]
        loss_eval = params["loss_test"]


        if horizon is True:

            # Specify the number of features and the stride for generating timeseries raw_data
            num_features = 30
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            losses = []

            for output_window in x:

                test_dataset = TimeSeriesLSTMnp(data.permute(1, 0),
                                                input_window,
                                                output_window)

                test_dataloader = DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=True)

                # Specify the device to be used for testing
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize the model and load the saved state dict
                model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)
                model.load_state_dict(saved_model["model_state_dict"])
                model.to(device)

                loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
                print("Output window: {}, Loss: {}".format(output_window, loss))
                losses.append(loss)
                wandb.log({"Horizon": output_window, "Test Loss": loss})

            loss_list.append((losses, model_num[m][1]))
        loss_list_eval.append((loss_eval, model_num[m][1]))
        wandb.finish()

    if horizon is True: plot_loss_horizon(loss_list, loss_type, id)
    plot_loss_combined(loss_list_eval, id, loss_type)




if __name__ == "__main__":
    main()
