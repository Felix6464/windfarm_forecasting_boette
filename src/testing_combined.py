from models.LSTM_enc_dec import *
from plots import *
from data_preprocessing import *
from utility_functions import *
import torch.utils.data as datat
from torch.utils.data import DataLoader
import torch.nn as nn



def main():

    data = pd.read_csv("preprocessed_data/filtered_dataset_brazil2_time_lag_corr_144.csv")
    print("Data shape : {}".format(data.shape))

    data = np.array(data).T
    data = torch.from_numpy(data)

    # Calculate the mean and standard deviation along the feature dimension
    data = normalize_data(data)

    # Specify the model number of the model to be tested
    #britain 144-144
    model_num_lstm_base = "8763179np"
    model_num_lstm = "2383946np"
    model_num_gru = "9411715np"
    model_num_lstm_input = "7540309np"
    model_num_lstm_input_tf = "3365362np"

    #brazil 144-144
    model_num_lstm_base = "3944041np"
    model_num_lstm = "1571176np"
    model_num_gru = "1902029np"
    model_num_lstm_input = "5308320np"
    model_num_lstm_input_tf = "1232841np"

    # Specify the number of features and the stride for generating timeseries raw_data
    input_window = 144
    batch_size = 256
    loss_type = "RMSE"

    model_lstm_base, model_lstm, model_lstm_inp, model_gru, model_lstm_inp_tf = load_models_testing(model_num_lstm_base,
                                                                                            model_num_lstm,
                                                                                            model_num_lstm_input,
                                                                                            model_num_gru,
                                                                                            model_num_lstm_input_tf)


    loss_list = []
    loss_list_temp = []


    x = range(1, 145)

    for output_window in x:
        print("Output window : {}".format(output_window))

        test_dataset = TimeSeriesLSTMnp(data.permute(1, 0),
                                        input_window,
                                        output_window)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True)


        loss_gru = model_gru.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_base = model_lstm_base.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm = model_lstm.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_inp = model_lstm_inp.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
        loss_lstm_inp_tf = model_lstm_inp_tf.evaluate_model(test_dataloader, output_window, batch_size, loss_type)

        loss_list_temp.append([loss_gru, loss_lstm_base, loss_lstm, loss_lstm_inp, loss_lstm_inp_tf])



    loss_list.append(([lst[0] for lst in loss_list_temp], f"{'GRU'}"))
    loss_list.append(([lst[1] for lst in loss_list_temp], f"{'LSTM-Base'}"))
    loss_list.append(([lst[2] for lst in loss_list_temp], f"{'LSTM-Enc-Dec'}"))
    loss_list.append(([lst[3] for lst in loss_list_temp], f"{'LSTM-Enc-Dec-Input'}"))
    loss_list.append(([lst[4] for lst in loss_list_temp], f"{'LSTM-Enc-Dec-Input-TF'}"))

    model_nums = str([model_num_gru, model_num_lstm_base, model_num_lstm, model_num_lstm_input, model_num_lstm_input_tf])
    plot_loss_horizon_combined(loss_list, model_nums, loss_type)



if __name__ == "__main__":
    main()