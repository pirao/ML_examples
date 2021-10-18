import pickle
from scipy.sparse.construct import rand

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class iforest():
    def __init__(self, data):
        """Object used to create the Isolation Forest model

        Args:
            data (pd): Pandas dataframe of the dataset
        """
        self.X = data.copy()
        self.model = None

    def create_model(self, **kwargs):
        """ Method used to create isolation forest
        
        **kwargs: Parâmeters of object IsolationForest from sklearn.ensemble.
        """
        self.model = IsolationForest(max_samples=1.0, n_jobs=-1, **kwargs)
        self.model.fit(self.X)
        return

    def clustering(self):
        """Apply the clustering model to the dataset

        Returns:
            pd: pandas dataframe with a new column containing the cluster label "GMM_cluster" 
        """
        if not self.model:
            print("Create a model with the method create_model first")

        self.Y_ = self.model.predict(self.X)
        cluster_data = self.X.copy()
        cluster_data['GMM_cluster'] = self.Y_

        return cluster_data

    def save_model(self, path):
        """Save the model inside a specific path

        Args:
            path (string): Folder path to save the model
        """
        pickle.dump(self.model, open(path, 'wb'))
        return

    def load_model(self, path):
        """Load a previously trained model from a specific path

        Args:
            path (string): Folder path to load the model from
        """
        self.model = pickle.load(open(path, 'rb'))
        return
