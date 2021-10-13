import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from yellowbrick.cluster import silhouette_visualizer


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
        kl = KneeLocator(self.n_range, self.ine,curve="convex", direction='decreasing')
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
