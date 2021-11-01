import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.visualization import plot_intermediate_values
from optuna.integration import KerasPruningCallback

import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import tensorflow as tf

from keras.utils.vis_utils import plot_model


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#################################################
#  Plotting learning curve
##################################################


def plot_learning_curve(history):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'],label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    plt.show()
    return


##################################################
# Vanilla NN - Regression
##################################################

class optuna_vanilla_NN:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.best_model = None
        self._model = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.callbacks = None
        self.history = None
        self._history = None

    def objective(self, trial):

        MAX_EPOCHS = 500
        PATIENCE = 12       # Number of epochs to check if the error continues to decrease - early stopping
        INTERVAL = 125      # Intermediate results saved and pruned/removed if it is bad

        tf.keras.backend.clear_session()

        # 2. Suggest values of the hyperparameters using a trial object.
        n_layers = 3  # trial.suggest_int('n_layers', 3,4)

        # 3. Create the model
        model = tf.keras.Sequential()

        for i in range(n_layers):
            num_neurons = trial.suggest_categorical('n_units_l{}'.format(i), list(np.arange(300, 1500, 25, dtype=int)))
            activation = trial.suggest_categorical('activ_fun_l{}'.format(i), ['linear', 'relu', 'tanh'])  # linear for 3 layers wasnt good
            model.add(tf.keras.layers.Dense(num_neurons, activation=activation))
            model.add(tf.keras.layers.BatchNormalization())

        # Output layer is linear for regression problems and size 1 because I want a single output
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        # 4. Defining callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=PATIENCE,
                                                          mode='min')

        optuna_pruner = KerasPruningCallback(trial, "val_loss", interval=INTERVAL)
        self.callbacks = [early_stopping, optuna_pruner]

        # 5. Compile and fit the model
        learning_rate = 1e-3  # trial.suggest_loguniform("lr", 1e-4, 1e-2)
        model.compile(loss=tf.losses.MeanSquaredError(),optimizer=tf.optimizers.Adam(learning_rate=learning_rate))

        self._history = model.fit(self.X_train,
                                  self.y_train,
                                  epochs=MAX_EPOCHS,
                                  validation_split=0.15,
                                  callbacks=self.callbacks,
                                  shuffle=True,
                                  verbose=0)

        self._model = model

        MSE_score_test_set = model.evaluate(self.X_test, self.y_test, verbose=0)
        return MSE_score_test_set

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self._model
            self.history = self._history

    def save_model(self,path):
        self.best_model.save(path)
    
    def load_model(self,path):
        model = tf.keras.models.load_model(path)
        return model
    
    def plot_model(self, filepath='Images/Vanilla_NN_plot.png'):
        plot_model(self.best_model, to_file=filepath, show_shapes=True, show_layer_names=True)


##################################################
# 1D autoencoder
##################################################
