from models.LSTM_enc_dec_input import *
from plots import *
from utility_functions import *
import torch.utils.data as datat
from torch.utils.data import DataLoader
from data_preprocessing import normalize_data


def main():

    # Load the preprocessed data
    data = pd.read_csv("./preprocessed_data/filtered_concatenated_data_2019_2021.csv")
    data = np.array(data).T
    data = torch.from_numpy(data)
    print("Data shape : {}".format(data.shape))

    # Calculate the mean and standard deviation along the feature dimension
    data = normalize_data(data)

    # Split the data into training and testing
    index_train = int(0.9 * len(data[0, :]))
    data = data[:, index_train:]

    # Specify the model number to be used for testing
    # Britain windfarm models
    model_num = [("8143199np", "2-1"),
                 ("3183351np", "6-6"),
                 ("8092051np", "36-36"),
                 ("8300243np", "144-144")]
    
    #144-144 - own selected features min-max normalized and not
    model_num = [("8300243np", "own_select"),
                 ("8940283np", "own_select_minmax")]

    
    # Brazil windfarm models 
    model_num = [("4459175np", "2-1"),
                 ("4909119np", "6-6"),
                 ("2248183np", "36-36"),
                 ("8294622np", "144-144")]
    
        #144-144 - global model approach (double data over 2 years) and normal approach over 1 year
    model_num = [("8294622np", "global"),
                 ("2383946np", "standard")]
    
    # on global data 
    model_num = [("4119988np", "288-288"),
                 ("9522982np", "77-144"),
                 ("8124366np", "288-144"),
                 ("3932602np", "144-144")]

    id = ["horizon_britain_benchmark"]

    model_num = [("3432090np", "144-144")]

    loss_list = []
    loss_list_eval = []
    horizon = True

    for m in range(len(model_num)):
        saved_model = torch.load(f"./final_models/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]
        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        input_window = params["input_window"]
        batch_size = params["batch_size"]
        loss_type = params["loss_type"]
        shuffle = params["shuffle"]
        loss_eval = params["loss_test"]


        if horizon is True:

            # Specify the number of features and the stride for generating timeseries raw_data
            num_features = 11
            x = range(1, 145)
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
                model = LSTM_Sequence_Prediction_Input(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)
                model.load_state_dict(saved_model["model_state_dict"])
                model.to(device)

                loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
                print("Output window: {}, Loss: {}".format(output_window, loss))
                losses.append(loss)

            loss_list.append((losses, model_num[m][1]))
        loss_list_eval.append((loss_eval, model_num[m][1]))
        wandb.finish()

    if horizon is True: plot_loss_horizon(loss_list, loss_type, id)


if __name__ == "__main__":
    main()
