import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from numpy import random



def plot_model_forecast_PC(lstm_model, train_data, target_data, test_data, test_target, rand, num_rows=5):
    """
    Plot examples of the LSTM encoder-decoder evaluated on the training/test raw_data.

    Args:
        lstm_model (LSTM): Trained LSTM encoder-decoder model.
        train_data (np.array): Windowed training input raw_data.
        target_data (np.array): Windowed training target raw_data.
        test_data (np.array): Windowed test input raw_data.
        test_target (np.array): Windowed test target raw_data.
        rand: Identifier.
        num_rows (int): Number of training/test examples to plot.

    Returns:
        None
    """

    train_data = train_data.detach().cpu()
    target_data = target_data.detach().cpu()
    test_data = test_data.detach().cpu()
    test_target = test_target.detach().cpu()

    print("Xtrain.shape: ", train_data.shape)
    print("Ytrain.shape: ", test_target.shape)

    # Input nd output window size
    input_window = train_data.shape[1]
    output_window = test_target.shape[1]

    fig, ax = plt.subplots(num_rows, 2, figsize=(13, 15))

    # Plot training/test predictions for a manually set index i (for better visualization)
    i = 0

    # Plot training/test predictions
    for row in range(num_rows):
        # Train set
        i += 20

        x_train = train_data[row+i, :, :]
        y_train_pred = lstm_model.predict(x_train, target_len=output_window, prediction_type="forecast").cpu()
        ax[row, 0].plot(np.arange(0, input_window), train_data[row+i, :, 0].cpu(), 'k', linewidth=2, label='Input')
        ax[row, 0].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[train_data[row+i, -1, 0]], target_data[row+i, :, 0]]),
                        color="blue", linewidth=2, label='Target')
        ax[row, 0].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[train_data[row+i, -1, 0]], y_train_pred[0, :, 0]]),
                        color="red", linewidth=2, label='Prediction')
        ax[row, 0].set_xlim([0, input_window + output_window - 1])
        ax[row, 0].set_xlabel('$Timestep$', fontsize=15)
        ax[row, 0].set_ylabel('$Prediction Value$', fontsize=15)

        # Test set
        x_test = test_data[row+i, :, :]
        y_test_pred = lstm_model.predict(x_test, target_len=output_window, prediction_type="forecast").cpu()
        ax[row, 1].plot(np.arange(0, input_window), test_data[row+i, :, 0], 'k', linewidth=2, label='Input')
        ax[row, 1].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[test_data[row+i, -1, 0]], test_target[row+i, :, 0]]),
                        color="blue", linewidth=2, label='Target')
        ax[row, 1].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[test_data[row+i, -1, 0]], y_test_pred[0, :, 0]]),
                        color="red", linewidth=2, label='Prediction')
        ax[row, 1].set_xlim([0, input_window + output_window - 1])
        ax[row, 1].set_xlabel('$Timestep$', fontsize=15)
        ax[row, 1].set_ylabel('$Prediction Values$', fontsize=15)

        if row == 0:
            ax[row, 0].set_title('Prediction on Train Data', fontsize=15)
            ax[row, 1].legend(bbox_to_anchor=(1, 1))
            ax[row, 1].set_title('Prediction on Test Data', fontsize=15)

    plt.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'final_plots_cluster/predictions_PC_{rand}_2.png')
    plt.show()
    plt.close()

    return



def plot_loss(loss_train, loss_eval, identifier, loss_type):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, 'b-', marker='o', markersize=5, linewidth=1, label='Loss-Train')
    plt.plot(epochs, loss_eval, color="r", marker='o', markersize=5, linewidth=1, label='Loss-Eval')
    plt.title(f'{loss_type} per Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'final_plots_cluster/loss_{loss_type}_combined_{identifier}.png', dpi=300)
    plt.show()


def plot_loss_combined(loss_values, identifier, loss_type):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_values[0][0][:50]) + 1)
    for m in range(len(loss_values)):
        loss = loss_values[m][0][:50]
        identifier = loss_values[m][1]
        plt.plot(epochs, loss, c=random.rand(3,), marker='o', markersize=5, linewidth=1, label=f'Loss-{identifier}')
    plt.title(f'{loss_type} per Epoch on Validation Set', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Eval Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'final_plots/loss_{loss_type}_combined_{identifier}_.png', dpi=300)
    plt.show()

def plot_loss_horizon_combined(loss_values, identifier, loss_type, tau=None):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_values[0][0]) + 1)
    for m in range(len(loss_values)):
        loss = loss_values[m][0]
        id = loss_values[m][1]
        if tau is not None: 
            loss = [loss[tau[0]], loss[tau[1]], loss[tau[2]]]
            epochs = [tau[0]+1, tau[1]+1, tau[2]+1]
        plt.plot(epochs, loss, c=random.rand(3,), marker='o', markersize=5, linewidth=1, label=f'Loss-{id}')
    plt.title(f'{loss_type} per Horizon Length on Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'final_plots_cluster/xloss_{loss_type}_horizon_combined_{identifier}_{tau}.png', dpi=300)
    plt.show()

def plot_loss_horizon(loss_values, loss_type, id, tau=None):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_values[0][0]) + 1)
    for m in range(len(loss_values)):
        loss = loss_values[m][0]
        identifier = loss_values[m][1]
        if tau is not None: 
            loss = [loss[tau[0]], loss[tau[1]], loss[tau[2]]]
            epochs = [tau[0]+1, tau[1]+1, tau[2]+1]
        plt.plot(epochs, loss, c=random.rand(3,), marker='o', markersize=5, linewidth=1, label=f'Loss-{identifier}')
    plt.title(f'{loss_type} per Horizon Length on Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xticks(epochs, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'final_plots_cluster/loss_{loss_type}_horizon_{id}_{tau}_x.png', dpi=300)
    plt.show()

def plot_loss_horizon_spread(loss_values, loss_type, id, tau=None):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    loss_values_np = np.array([loss_values[m][0] for m in range(len(loss_values))])
    loss_mean = loss_values_np.mean(axis=0)
    loss_std = np.std(loss_values_np, axis=0)
    loss_max = loss_mean + loss_std
    loss_min = loss_mean - loss_std
    identifier = loss_values[0][1]
    epochs = range(1, len(loss_values[0][0]) + 1)
    if tau is not None: 
        loss_mean = [loss_mean[tau[0]], loss_mean[tau[1]], loss_mean[tau[2]]]
        loss_min = [loss_min[tau[0]], loss_min[tau[1]], loss_min[tau[2]]]
        loss_max = [loss_max[tau[0]], loss_max[tau[1]], loss_max[tau[2]]]
        epochs = [tau[0]+1, tau[1]+1, tau[2]+1]
    plt.plot(epochs, loss_mean, c="b", marker='o', markersize=5, linewidth=1, label=f'Loss-{identifier}')
    plt.fill_between(epochs, loss_max, loss_min, color='r', alpha=0.5, label='model std')
    plt.title(f'{loss_type} per Horizon Length on Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xticks(epochs, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'final_plots_cluster/loss_{loss_type}_horizon_{id}_spread.png', dpi=300)
    plt.show()


def plot_correlation_heatmap(correlation_matrix):
    """
    Plot a correlation heatmap based on a correlation matrix.

    Args:
        correlation_matrix (pandas.Series): Pandas Series representing the correlation matrix.

    Returns:
        None
    """
    # Convert the correlation series to a DataFrame
    correlation_df = correlation_matrix.unstack().reset_index()
    correlation_df.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # Pivot the DataFrame to create a correlation matrix
    correlation_matrix = correlation_df.pivot('Feature 1', 'Feature 2', 'Correlation')

    # Plot the correlation heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()