import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from yellowbrick.cluster import silhouette_visualizer
from sklearn.neighbors import NearestNeighbors

class kmeans():
    def __init__(self, df, cat=False):

        self.X = df.copy()
        self.model = None

    def cluster_criterion(self, n_clusters, verbose=False, normalize=True):

        ine = []
        sil = []
        calinski_harabasz = []
        davies_bouldin = []

        n_range = range(2, n_clusters + 1)
        self.n_range = n_range

        for i in tqdm(n_range):
            kmeans = KMeans(n_clusters=i, max_iter=5000,random_state=42, init='k-means++')
            kmeans.fit(self.X)

            # Elbow
            ine.append(kmeans.inertia_)
            sil.append(silhouette_score(self.X, kmeans.predict(self.X)))                              # Maximize
            calinski_harabasz.append(calinski_harabasz_score(self.X, kmeans.predict(self.X)))         # Maximize
            davies_bouldin.append(davies_bouldin_score(self.X, kmeans.predict(self.X)))               # Minimize

            if verbose:
                print('')
                print(f'Number of groups: {i} --- Inertia: {ine[i-2]} --- Silhouette coefficient: {sil[i-2]}')
                print(f'Number of groups: {i} --- Inertia: {ine[i-2]} --- calinski_harabasz: {calinski_harabasz[i-2]}')
                print(f'Number of groups: {i} --- Inertia: {ine[i-2]} --- davies_bouldin: {davies_bouldin[i-2]}')
                print('')

        # Converting to numpy arrays
        self.ine = np.array(ine)
        self.sil = np.array(sil)
        self.calinski_harabasz = np.array(calinski_harabasz)
        self.davies_bouldin = np.array(davies_bouldin)

        if normalize:
            self.ine = (self.ine-self.ine.min()) / (self.ine.max()-self.ine.min())
            self.sil = (self.sil-self.sil.min()) / (self.sil.max()-self.sil.min())
            self.calinski_harabasz = (self.calinski_harabasz-self.calinski_harabasz.min())/(self.calinski_harabasz.max()-self.calinski_harabasz.min())
            self.davies_bouldin = (self.davies_bouldin-self.davies_bouldin.min())/(self.davies_bouldin.max()-self.davies_bouldin.min())

        # Index of optimal metric value
        kl = KneeLocator(self.n_range, self.ine, curve="convex", direction='decreasing', S=2)
        self.knee = kl.knee
        self.sil_index = np.where(self.sil == np.amax(self.sil))[0][0]
        self.cal_index = np.where(self.calinski_harabasz == np.amax(self.calinski_harabasz))[0][0]
        self.dav_index = np.where(self.davies_bouldin == np.amin(self.davies_bouldin))[0][0]

        # Plotting
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(26, 13))

        ax[0, 0].plot(self.n_range, self.ine)
        ax[0, 0].set_xticks(self.n_range)
        ax[0, 0].set_ylabel("Inertia")
        ax[0, 0].axvline(x=n_range[self.knee],ymax=ine[self.knee], color='red', linestyle='--')

        ax[0, 1].plot(self.n_range, self.sil)
        ax[0, 1].set_xticks(self.n_range)
        ax[0, 1].set_ylabel("Silhouette coefficient")
        ax[0, 1].set_xlabel("Number of groups")
        ax[0, 1].axvline(x=n_range[self.sil_index],ymax=self.sil.max(), color='red', linestyle='--')

        ax[1, 0].plot(self.n_range, self.calinski_harabasz)
        ax[1, 0].set_xticks(self.n_range)
        ax[1, 0].set_ylabel("Calinski harabasz score")
        ax[1, 0].set_xlabel("Number of groups")
        ax[1, 0].axvline(x=n_range[self.cal_index], ymax=self.calinski_harabasz.max(), color='red', linestyle='--')

        ax[1, 1].plot(self.n_range, self.davies_bouldin)
        ax[1, 1].set_xticks(self.n_range)
        ax[1, 1].set_xlabel("Number of groups")
        ax[1, 1].set_ylabel("Davies bouldin score")
        ax[1, 1].axvline(x=n_range[self.dav_index],ymax=self.davies_bouldin.max(), color='red', linestyle='--')

        plt.tight_layout()
        plt.show()

    def create_model(self, n_clusters):
        self.model = KMeans(n_clusters=n_clusters,max_iter=5000, random_state=42)
        self.model.fit(self.X)
        return

    def silhouette_plot(self):
        if not self.model:
            print("Create a clustering model first")

        silhouette_visualizer(self.model, self.X)
        return

    def clustering(self):
        if not self.model:
            print("Create a clustering model first")

        self.Y_ = self.model.predict(self.X)
        cluster_data = self.X.copy()
        cluster_data['cluster_kmeans'] = self.Y_

        return cluster_data

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))
        return

    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))
        return

####################################
# DBSCAN
#####################################


class dbscan():
    def __init__(self, data):

        self.X = data.copy()
        self.model = None

    def plot_baseline_eps(self):

        nbrs = NearestNeighbors(n_neighbors=3).fit(self.X)
        dists, inds = nbrs.kneighbors(self.X)

        # Sort in ascending order
        dists = np.sort(dists, axis=0)
        # Get the distance of the k'th farthest point
        dists = dists[:, -1]

        n_range = np.arange(0, len(dists), 1)
        kl = KneeLocator(n_range, dists, curve="convex",direction='increasing', S=3)
        #kl.plot_knee()
        self.knee = kl.knee

        # Plot baseline eps
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(dists)
        ax.axhline(xmin=0, xmax=len(dists),y=dists[self.knee], color='red', linestyle='--')

        ax.set_xlabel("Ascending data index")
        ax.set_ylabel("Minimum intersample distance")
        ax.set_title('Baseline eps value for 3 nearest neighbors')
        ax.legend(['eps values', 'threshold: ' +str(round(dists[self.knee], 4))])
        plt.show()
        return

    def plot_hyperparameters(self, nrows, ncols, eps_list=[0.026, 0.029, 0.032], min_samples_list=[10, 15, 20], figsize=(15, 20), remove_outliers=True):

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,tight_layout=True, sharey=True, sharex=True)
        ax = ax.reshape((nrows*ncols,))

        index = 0
        for eps in tqdm(eps_list):
            for min_sample in min_samples_list:
                dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(self.X)
                self.Y_ = dbscan.labels_
                cluster_data = self.X.copy()
                cluster_data['cluster_dbscan'] = self.Y_

                if remove_outliers:
                    df_in = cluster_data.loc[(
                        cluster_data['cluster_dbscan'] != -1)]
                    sns.scatterplot(data=df_in, x='PC1', y='PC2', hue='cluster_dbscan',
                                    ax=ax[index], palette='Set1', legend=False)

                else:
                    sns.scatterplot(data=cluster_data, x='PC1', y='PC2',
                                    hue='cluster_dbscan', ax=ax[index], palette='Set1', legend=False)

                ax[index].set_title('eps = ' + str(eps) +
                                    ', min sample = ' + str(min_sample))
                index = index + 1
        return

    def create_model(self, eps, msamples=5):
        self.model = DBSCAN(eps=eps, min_samples=msamples, n_jobs=-1)
        self.model.fit(self.X)
        return

    def clustering(self):

        if not self.model:
            print("Create a model first using the create model method ")

        self.Y_ = self.model.labels_
        cluster_data = self.X.copy()
        cluster_data['cluster_dbscan'] = self.Y_
        return cluster_data

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))
        return

    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))
        return


##############################
# Gaussian mixture model
##############################

class GMM():
    def __init__(self, data):
        '''
        Objeto que cria um modelo GMM.
        I/O:
            data: conjunto de dados contendo os dados do VI.
        '''
        self.X = data.copy()
        self.model = None

    def BIC_criterion(self, n_clusters, gradient=False, verbose=False):
        '''
        Método que verifica o critério de BIC para definição no número de grupos ideal para o agrupamento.
        I/O:
            n_clusters: inteiro que indica o número máximo de clusters participantes do critério;
            gradient: booleano que indica quando a curva plotada deve ser a curva gradiente do BIC ou não;
            verbose: booleano que indica se deve ser informado, ou não, o andamento do cálculo do BIC.
        '''
        bic = []
        n_range = range(1, n_clusters + 1)
        cv_types = ['spherical', 'tied', 'diag', 'full']

        for cv_type in tqdm(cv_types):
            type_bic = []
            for i in n_range:
                gmm = GaussianMixture(
                    n_components=i, covariance_type=cv_type, max_iter=5000, random_state=42)
                gmm.fit(self.X)
                temp_bic = gmm.bic(self.X)
                type_bic.append(temp_bic)
                if verbose:
                    print(
                        f'Tipo da covariância: {cv_type} --- Número de agrupamentos: {i} --- BIC: {temp_bic}')
            bic.append(type_bic)
        bic = np.array(bic)

        if gradient:
            bic = np.array([np.gradient(array) for array in bic])

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(cv_types)):
            ax.plot(n_range, bic[i, :], label=cv_types[i])
        ax.set_xticks(n_range)
        ax.legend()
        ax.set_xlabel("Número de agrupamentos")
        ax.set_ylabel("BIC")

        plt.show()

    def create_model(self, cv_type, n_clusters):
        '''
        Método que cria o modelo de misturas gausianas.
        I/O:
            cv_type: string que indica o tipo de covariança usada para a definição do modelo ("spherical", "tied", "diag" ou "full");
            n_cluster: inteiro que indica o número de grupos existentes no conjunto de dados.
        '''
        self.model = GaussianMixture(
            n_components=n_clusters, covariance_type=cv_type, max_iter=5000, random_state=42)
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
