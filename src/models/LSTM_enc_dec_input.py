from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn
import wandb



class TimeSeriesLSTM(Dataset):
    def __init__(self, xarr, input_window, output_window, one_hot_month=False):
        self.input_window = input_window
        self.output_window = output_window
        self.xarr = xarr.compute()

    def __len__(self):
        return len(self.xarr['time']) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.xarr.isel(time=slice(idx, idx+self.input_window))
        target = self.xarr.isel(time=slice(idx+self.input_window, idx+self.input_window  + self.output_window))

        # One hot encoding of month
        idx_month = input.isel(time=-1).time.dt.month.astype(int) - 1
        one_hot_month = np.zeros(12)
        one_hot_month[idx_month] = 1
        one_hot_month = torch.from_numpy(one_hot_month).float()

        input = torch.from_numpy(input.data).float()
        if one_hot_month is True:
            target = torch.from_numpy(target.data[np.newaxis]).float()
        else:
            target = torch.from_numpy(target.data).float()

        label = {
            'idx_input': torch.arange(idx, idx+self.input_window),
            'idx_target': torch.arange(idx+self.input_window, idx+self.input_window  + self.output_window),
            'month': one_hot_month
        }

        return input, target, label


class TimeSeriesLSTMnp(Dataset):
    def __init__(self, arr, input_window, output_window):
        self.input_window = input_window
        self.output_window = output_window
        self.arr = arr

    def __len__(self):
        return len(self.arr[:, 0]) - self.input_window - self.output_window - 2


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.arr[idx:idx+self.input_window, :].float()
        target = self.arr[idx+self.input_window:idx+self.input_window  + self.output_window, :].float()

        label = "not set"

        return input, target, label




class LSTM_Encoder(nn.Module):
    """
    Encodes time-series sequence
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstms = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout))


    def forward(self, x_input, encoder_hidden):
        """
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        """

        for i in range(self.num_layers):
            lstm_out, hidden = self.lstms[i](x_input, encoder_hidden)
            x_input, encoder_hidden = lstm_out, hidden
        return lstm_out, hidden

    def init_hidden(self, batch_size):
        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class LSTM_Decoder(nn.Module):
    """
    Decodes hidden state output by encoder
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(LSTM_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstms = nn.ModuleList()

        for i in range(self.num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.lstms.append(nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout))

        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, decoder_input, decoder_hidden, outputs=None, target_batch=None, training_prediction=None, target_len=None, teacher_forcing_ratio=None, prediction_type=None):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence
        '''

        decoder_input = decoder_input.unsqueeze(1)


        if training_prediction == 'teacher_forcing':
            # Use teacher forcing
            if random.random() < teacher_forcing_ratio:
                for t in range(target_len):
                    for i in range(self.num_layers):
                        lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                        decoder_input = decoder_hidden[0].permute(1, 0, 2)

                    decoder_output = self.linear(lstm_out.squeeze(0))
                    outputs[:, t, :] = decoder_output[:, 0, :]
                    decoder_input = target_batch[:, t, :]

            # Predict recursively
            else:
                for t in range(target_len):
                    for i in range(self.num_layers):
                        lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                        decoder_input = decoder_hidden[0].view(decoder_hidden[0].shape[1], 1, decoder_hidden[0].shape[2])

                    decoder_output = self.linear(lstm_out.squeeze(0))
                    outputs[:, t, :] = decoder_output
                    decoder_input = decoder_output

        elif training_prediction == 'mixed_teacher_forcing':
            # Predict using mixed teacher forcing
            for t in range(target_len):
                for i in range(self.num_layers):
                    lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                    decoder_output = self.linear(lstm_out.squeeze(0))
                    decoder_input = decoder_hidden[0].permute(1, 0, 2)


                outputs[:, t, :] = decoder_output[:, 0, :]

                # Predict with teacher forcing
                if random.random() < teacher_forcing_ratio:
                    decoder_input = target_batch[:, t, :].unsqueeze(1)

                # Predict recursively
                else:
                    decoder_input = decoder_output

        else:
            # Predict recursively
            for t in range(target_len):
                for i in range(self.num_layers):
                    lstm_out, decoder_hidden = self.lstms[i](decoder_input, decoder_hidden)
                    decoder_output = self.linear(lstm_out.squeeze(0))
                    decoder_input = decoder_hidden[0].permute(1, 0, 2)
                #print(decoder_output.shape)
                outputs[:, t, :] = decoder_output[:, 0, :]
                decoder_input = decoder_output


        return outputs, decoder_hidden




class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class LSTM_Sequence_Prediction_Input(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''

        super(LSTM_Sequence_Prediction_Input, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = LSTM_Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = LSTM_Decoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def train_model(self, train_dataloader, eval_dataloader,optimizer, config):
        """
        Train an LSTM encoder-decoder model.

        :param input_len:
        :param target_test:
        :param input_test:
        :param input_tensor:              Input raw_data with shape (seq_len, # in batch, number features)
        :param target_tensor:             Target raw_data with shape (seq_len, # in batch, number features)
        :param n_epochs:                  Number of epochs
        :param target_len:                Number of values to predict
        :param batch_size:                Number of samples per gradient update
        :param training_prediction:       Type of prediction to make during training ('recursive', 'teacher_forcing', or
                                          'mixed_teacher_forcing'); default is 'recursive'
        :param teacher_forcing_ratio:     Float [0, 1) indicating how much teacher forcing to use when
                                          training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
                                          number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
                                          Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
                                          teacher forcing.
        :param learning_rate:             Float >= 0; learning rate
        :param dynamic_tf:                dynamic teacher forcing reduces the amount of teacher forcing for each epoch
        :return losses:                   Array of loss function for each epoch
        """

        if config["wandb"] is True:
            wandb.init(project=f"ML-Windfarm-{config['model_label']}", config=config, name=config['name'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)

        # Initialize array to store losses for each epoch
        losses = np.full(config["num_epochs"], np.nan)
        losses_test = np.full(config["num_epochs"], np.nan)

        # Initialize optimizer and criterion
        if config["loss_type"] == 'MSE':
            criterion = nn.MSELoss()
        elif config["loss_type"] == 'L1':
            criterion = nn.L1Loss()
        elif config["loss_type"] == 'RMSE':
            criterion = RMSELoss()



        with trange(config["num_epochs"]) as tr:
            for epoch in tr:
                batch_loss = 0.0
                batch_loss_test = 0.0
                train_len = 0
                eval_len = 0

                for input, target, l in eval_dataloader:
                    eval_len += 1

                    input_eval, target_eval = input, target
                    input_eval = input_eval.to(device)
                    target_eval = target_eval.to(device)

                    with torch.no_grad():
                        self.eval()

                        Y_test_pred = self.predict(input_eval, config["output_window"])
                        Y_test_pred = Y_test_pred.to(device)
                        loss_test = criterion(Y_test_pred, target_eval)
                        batch_loss_test += loss_test.item()

                #batch_loss_test /= eval_len
                losses_test[epoch] = batch_loss_test

                for input_batch, target_batch, l in train_dataloader:
                    train_len += 1
                    self.train()

                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)


                    # Initialize outputs tensor
                    outputs = torch.zeros(config["batch_size"], config["output_window"], config["num_features"])
                    outputs = outputs.to(device)


                    # Zero the gradients
                    optimizer.zero_grad()

                    encoder_hidden = self.encoder.init_hidden(config["batch_size"])
                    encoder_hidden = (encoder_hidden[0].to(device), encoder_hidden[1].to(device))
                    # Encoder forward pass
                    encoder_output, encoder_hidden = self.encoder(input_batch, encoder_hidden)

                    # Decoder input for the current batch
                    decoder_input = input_batch[:, -1, :]

                    #print("decoder_hidden shape : {}".format(decoder_hidden.shape))

                    outputs, decoder_hidden = self.decoder(decoder_input,
                                                           encoder_hidden,
                                                           outputs=outputs,
                                                           training_prediction=config["training_prediction"],
                                                           target_len=config["output_window"],
                                                           teacher_forcing_ratio=config["teacher_forcing_ratio"],
                                                           target_batch=target_batch)


                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # Backpropagation and weight update
                    loss.backward()
                    optimizer.step()

                # Compute average loss for the epoch
                #batch_loss /= train_len
                losses[epoch] = batch_loss

                # Dynamic teacher forcing
                if config["dynamic_tf"] and config["teacher_forcing_ratio"] > 0:
                    config["teacher_forcing_ratio"] -= 0.01

                print("Epoch: {0:02d}, Training Loss: {1:.4f}, Test Loss: {2:.4f}".format(epoch, batch_loss, batch_loss_test))

                # Update progress bar with current loss
                tr.set_postfix(loss_test="{0:.3f}".format(batch_loss_test))
                wandb.log({"Epoch": epoch, "Training Loss": batch_loss, "Test Loss": batch_loss_test})
                wandb.watch(criterion, log="all")

        return losses, losses_test


    def evaluate_model(self, test_dataloader, target_len, batch_size, loss_type):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()

        eval_len = 0
        batch_loss_test = 0.0

        for input, target, l in test_dataloader:
            eval_len += 1
            self.eval()

            input_batch, target_batch = input, target
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                self.eval()

                Y_test_pred = self.predict(input_batch.float(), target_len)
                Y_test_pred = Y_test_pred.to(device)
                loss_test = criterion(Y_test_pred[:, -1, :], target_batch[:, -1, :])
                batch_loss_test += loss_test.item()

        #batch_loss_test /= eval_len


        return batch_loss_test


    def predict(self, input_tensor, target_len, prediction_type='test'):

        """
        : param input_tensor:      input raw_data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        # encode input_tensor
        if prediction_type == 'forecast':
            input_tensor = input_tensor.unsqueeze(0)  # add in batch size of 1

        encoder_hidden = self.encoder.init_hidden(input_tensor.shape[0])
        encoder_hidden = (encoder_hidden[0].to(device), encoder_hidden[1].to(device))

        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

        # Initialize outputs tensor
        outputs = torch.zeros(input_tensor.shape[0], target_len, input_tensor.shape[2], requires_grad=True)
        outputs = outputs.to(device)

        # decode input_tensor
        decoder_input = input_tensor[:, -1, :]

        outputs, decoder_hidden = self.decoder(decoder_input,
                                               encoder_hidden,
                                               outputs=outputs,
                                               target_len=target_len,
                                               prediction_type=prediction_type)

        if prediction_type == 'forecast':
            outputs = outputs.detach()


        return outputs

