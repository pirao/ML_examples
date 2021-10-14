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
        '''
        Objeto que cria um modelo de Isolation Forest.
        I/O:
            data: conjunto de dados contendo os dados do VI.
        '''
        self.X = data.copy()
        self.model = None

    def create_model(self, **kwargs):
        '''
        Método que cria o modelo de Isolation Forest.
        I/O:
            **kwargs: parâmetros do objeto IsolationForest de sklearn.ensemble.
        '''
        self.model = IsolationForest(max_samples=1.0, n_jobs=-1, **kwargs)
        self.model.fit(self.X)

    def clustering(self):
        '''
        Método que realiza o clustering do conjunto de dados trabalhado.
        I/O:
            return cluster_data: um pandas dataframe contendo o conjunto de dados com o rótulo dos agrupamentos feitos pelo modelo.
        '''
        if not self.model:
            print("Crie um modelo com o método create_model.")

        self.Y_ = self.model.predict(self.X)
        cluster_data = self.X.copy()
        cluster_data['cluster'] = self.Y_

        return cluster_data

    def save_model(self, path):
        '''
        Método que salva o modelo criado.
        I/O:
            path: string indicando o caminho do objeto referente ao modelo.
        '''
        pickle.dump(self.model, open(path, 'wb'))

    def load_model(self, path):
        '''
        Método que carrega um modelo previamente criado.
        I/O:
            path: string indicando o caminho do objeto referente ao modelo.
        '''
        self.model = pickle.load(open(path, 'rb'))
