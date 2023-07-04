import numpy as np
import random
from tqdm import trange
import torch
import torch.nn as nn
from torch import optim


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class LSTM_Sequence_Prediction(nn.Module):
    """
    train LSTM encoder-decoder and make predictions
    """

    def __init__(self, input_size, hidden_size, num_layers):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        '''

        super(LSTM_Sequence_Prediction, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = LSTM_Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = LSTM_Decoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def train_model(self, input_tensor, target_tensor, input_test, target_test, n_epochs, input_len, target_len, batch_size,
                    training_prediction, teacher_forcing_ratio, learning_rate, dynamic_tf, loss_type, optimizer):
        """
        Train an LSTM encoder-decoder model.

        :param input_len:
        :param target_test:
        :param input_test:
        :param input_tensor:              Input data with shape (seq_len, # in batch, number features)
        :param target_tensor:             Target data with shape (seq_len, # in batch, number features)
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        input_test = input_test.to(device)
        target_test = target_test.to(device)


        # Initialize array to store losses for each epoch
        losses = np.full(n_epochs, np.nan)
        losses_test = np.full(n_epochs, np.nan)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()


        #train_loader = DataLoader(dataset=list(zip(input_tensor, target_tensor)), batch_size=batch_size, shuffle=False)
        #test_loader = DataLoader(dataset=list(zip(input_test, target_test)), batch_size=batch_size, shuffle=False)

        # Calculate the number of batch iterations
        n_batches = input_tensor.shape[1] // batch_size
        n_batches_test = input_test.shape[1] // batch_size
        num_batch = n_batches
        num_batch_test = n_batches_test
        print("Number of batches : {}".format(n_batches))
        print("Number of batches test : {}".format(n_batches_test))

        n_batches = list(range(n_batches))
        n_batches_test = list(range(n_batches_test))

        with trange(n_epochs) as tr:
            for epoch in tr:
                random.shuffle(n_batches)
                random.shuffle(n_batches_test)
                batch_loss = 0.0
                batch_loss_test = 0.0

                for batch_idx in n_batches:
                #for batch_idx, data in enumerate(train_loader):
                    self.train()

                    #input_batch, target_batch = data
                    #input_batch = input_batch.to(device)
                    #target_batch = target_batch.to(device)
                    #print("Input batch : {} + shape : {} ".format(input_batch, input_batch.shape))
                    #print("Target batch : {} + shape : {}".format(targets, targets.shape))

                    # Select data for the current batch
                    input_batch = input_tensor[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                    target_batch = target_tensor[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]

                    # Initialize outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])
                    outputs = outputs.to(device)

                    # Initialize hidden state for the encoder
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Encoder forward pass
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # Decoder input for the current batch
                    decoder_input = input_batch[-1, :, :]
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # Predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # Use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # Predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # Predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output

                            # Predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]

                            # Predict recursively
                            else:
                                decoder_input = decoder_output


                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # Backpropagation and weight update
                    loss.backward()
                    optimizer.step()

                # Compute average loss for the epoch
                batch_loss /= num_batch
                losses[epoch] = batch_loss

                # Dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio -= 0.01

                #for batch_idx, data in enumerate(test_loader):
                for batch_idx in n_batches_test:
                    #input_test, target_test = data
                    #input_test = input_test.to(device)
                    #target_test = target_test.to(device)

                    with torch.no_grad():
                        self.eval()

                        # Select data for the current batch
                        input_test_batch = input_test[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                        target_test_batch = target_test[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                        input_test_batch = input_test_batch.to(device)
                        target_test_batch = target_test_batch.to(device)
                        #print("Input test shape : {} ".format(input_test.shape))
                        #print("Target test shape : {}".format(target_test.shape))
                        #print("Input test : {} ".format(input_test[:, input_len, :]))
                        #print("Input test shape : {} ".format(input_test[:, input_len, :].shape))


                        Y_test_pred = self.predict(input_test_batch, target_len=target_len)
                        Y_test_pred = Y_test_pred.to(device)
                        loss_test = criterion(Y_test_pred, target_test_batch)
                        batch_loss_test += loss_test.item()

                batch_loss_test /= num_batch_test
                losses_test[epoch] = batch_loss_test

                # Update progress bar with current loss
                tr.set_postfix(loss_test="{0:.3f}".format(batch_loss_test))

            return losses, losses_test


    def evaluate_model(self, input_test, target_test, target_len, batch_size, loss_type):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        input_test = input_test.to(device)
        target_test = target_test.to(device)

        # Initialize optimizer and criterion
        if loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif loss_type == 'L1':
            criterion = nn.L1Loss()
        elif loss_type == 'RMSE':
            criterion = RMSELoss()

        # Calculate the number of batch iterations
        n_batches_test = input_test.shape[1] // batch_size
        num_batch_test = n_batches_test
        n_batches_test = list(range(n_batches_test))

        random.shuffle(n_batches_test)
        batch_loss_test = 0.0

        for batch_idx in n_batches_test:

            with torch.no_grad():
                self.eval()

                # Select data for the current batch
                input_test_batch = input_test[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                target_test_batch = target_test[:, batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
                input_test_batch = input_test_batch.to(device)
                target_test_batch = target_test_batch.to(device)

                Y_test_pred = self.predict(input_test_batch, target_len=target_len)
                Y_test_pred = Y_test_pred.to(device)
                loss_test = criterion(Y_test_pred, target_test_batch)
                batch_loss_test += loss_test.item()

        batch_loss_test /= num_batch_test


        return batch_loss_test


    def predict(self, input_tensor, target_len, prediction_type='test'):

        """
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        # encode input_tensor
        if prediction_type == 'forecast':
            input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[1], input_tensor.shape[2])
        outputs = outputs.to(device)

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            if prediction_type == 'forecast':
                outputs[t] = decoder_output.squeeze(0)
            else:
                outputs[t] = decoder_output
            decoder_input = decoder_output

        if prediction_type == 'forecast':
            outputs = outputs.detach()


        return outputs

class LSTM_Encoder(nn.Module):
    """
    Encodes time-series sequence
    """

    def __init__(self, input_size, hidden_size, num_layers=3):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, x_input):
        """
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        """

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

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

    def __init__(self, input_size, hidden_size, num_layers=3):
        """
        : param input_size:     the number of features in the input_data
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers
        """

        super(LSTM_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence
        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden





def dataloader_seq2seq_feat(y, input_window, output_window, stride, num_features):
    '''
    Create a windowed dataset

    :param y:                Time series feature (array)
    :param input_window:     Number of y samples to give the model
    :param output_window:    Number of future y samples to predict
    :param stride:           Spacing between windows
    :param num_features:     Number of features
    :return X, Y:            Arrays with correct dimensions for LSTM
                             (i.e., [input/output window size # examples, # features])
    '''

    data_len = y.shape[1]
    num_samples = (data_len - input_window - output_window) // stride + 1

    # Initialize X and Y arrays with zeros
    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_features])

    for feature_idx in np.arange(num_features):
        for sample_idx in np.arange(num_samples):
            # Create input window
            start_x = stride * sample_idx
            end_x = start_x + input_window
            X[:, sample_idx, feature_idx] = y[feature_idx, start_x:end_x]

            # Create output window
            start_y = stride * sample_idx + input_window
            end_y = start_y + output_window
            Y[:, sample_idx, feature_idx] = y[feature_idx, start_y:end_y]

    return X, Y


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    : param Xtrain:                    windowed training input data (input window size, # examples, # features)
    : param Ytrain:                    windowed training target data (output window size, # examples, # features)
    : param Xtest:                     windowed test input data (input window size, # examples, # features)
    : param Ytest:                     windowed test target data (output window size, # examples, # features)
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

    '''

    X_train = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train, Y_train, X_test, Y_test