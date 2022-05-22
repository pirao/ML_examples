import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import false
from tqdm import tqdm
import pickle

has_rapids_env = True
try:
    from cuml.cluster import KMeans as cu_KMeans
    from cuml.cluster import DBSCAN as cu_DBSCAN
    from cuml.neighbors import NearestNeighbors as cu_NNeighbors
    import cupy as cp
except ModuleNotFoundError:
    has_rapids_env = False


from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
from yellowbrick.cluster import silhouette_visualizer
from sklearn.neighbors import NearestNeighbors


def NNeighbors(X, n_neighbors, use_cuda = False):
    
    if(use_cuda and has_rapids_env):
        
        nbrs = cu_NNeighbors(n_neighbors=n_neighbors).fit(X)
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    return nbrs.kneighbors(X)
    


def make_categorical(array):
    for i in range(len(array)):
        array[i] = 'outlier' if array[i] == -1 else 'cluster ' + str(array[i])

    return array


def my_range(begin, end, step):
    array = []
    while (round(begin, 3) < end):
        array += [begin]
        begin += step
    return array


def remove_outliers_(X, outlier_column, outlier_value=-1):
    """Remove the outliers of a given dataframe.

    Args:
        X (nd_dataframe): dataframe in which the outliers will be removed.
        outlier_column (string): column which contains the outlier values
        outlier_value (int, optional): Outlier value. Defaults to -1.

    Returns:
        nd_dataframe: returns the dataframe without the indexes marked as outliers.
    """
    return X.loc[(X[outlier_column] != outlier_value)]


def max_index(array):
    """Gets the index of the largets item of an given array.

    Args:
        array (list, numpy array): an list or numpy array

    Returns:
        integer: index of the largest item
    """
    if(has_rapids_env):
        aux = cp.array(array)
        return cp.where(aux == cp.amax(aux))[0][0]
    else:
        aux = np.array(array)
        return np.where(aux == np.amax(aux))[0][0]


def min_index(array):
    """Gets the index of the smallest item of an given array.

    Args:
        array (list, numpy array): an list or numpy array

    Returns:
        integer: index of the smallest item
    """
    if(has_rapids_env):
        aux = cp.array(array)
        return cp.where(aux == cp.amin(aux))[0][0]
    else:
        aux = np.array(array)
        return np.where(aux == np.amin(aux))[0][0]


def normalize_(array):
    """normalizes all values of an given array

    Args:
        array (list, numpy array): array to be normalized

    Returns:
        numpy array: normalized array
    """
    
    aux = cp.array(array) if (has_rapids_env) else np.array(array)
    return (aux - aux.min()) / (aux.max() - aux.min())


class _clustering():

    def __init__(self, df, cat=False):
        self.X = df.copy()
        self.model = None
        self.labels = None

    def get_labels(self):
        return [i for i in self.labels]

    def clustering(self, X):
        """Predict the labels for the data samples in X using the trained model.



        Args:
            X (array-like of shape (n_samples, n_features)): List of n_features-dimensional data points. Each row corresponds to a single data point.

        Returns:
            labels (array, shape (n_samples,): Component labels.
        """
        if not self.model:
            print("Create a clustering model first")

        else:
            self.labels = self.model.predict(X)
            cluster_data = self.X.copy()
            cluster_data['cluster'] = self.labels

        return cluster_data

    def get_data(self):
        """Get method for the class dataset

        Returns:
            array like (n_samples, n_features): returns the class dataset
        """
        return self.X

    def get_model(self):
        """Get method for the class model

        Returns:
            clustering model: returns the child respective clustering model
        """
        if not self.model:
            print("Create a clustering model first")
        return self.model

    def save_model(self, path):
        if not self.model:
            print("There is no model to save")
        pickle.dump(self.model, open(path, 'wb'))
        return

    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))
        return

    def silhouette_score_(self):
        try:
            return silhouette_score(self.X, self.labels)
        except ValueError:
            print("The number of clusters must be greater than 1!")
            return

    def calinski_harabasz_score_(self):
        try:
            return calinski_harabasz_score(self.X, self.labels)
        except ValueError:
            print("The number of clusters must be greater than 1!")
            return

    def davies_bouldin_score_(self):
        try:
            return davies_bouldin_score(self.X, self.labels)
        except ValueError:
            print("The number of clusters must be greater than 1!")
            return


class kmeans(_clustering):

    def __init__(self, data, use_cuda = False):
        super().__init__(data)
        if(use_cuda):
            if(has_rapids_env):
                self.use_cuda = use_cuda
            else:
                self.use_cuda = False;
                print("No rapids enviroments found.\nMoving to default KMeans instead.")
        else:
            self.use_cuda = False;

    def cluster_evaluation(self, n_clusters, verbose=False, normalize=True):

        if (n_clusters <= 2):
            print("n_clusters must be greater than 2.")
            return

        ine = []
        sil = []
        calinski_harabasz = []
        davies_bouldin = []

        n_range = range(2, n_clusters + 1)

        for i in tqdm(n_range):

            # creating, fitting and predicting the model
            km = kmeans(self.X)
            km.create_model(n_clusters=i)
            km.clustering(self.X)
  
            # computing the scores
            ine.append(km.inertia())  # Find Elbow
            sil.append(km.silhouette_score_())  # Maximize
            calinski_harabasz.append(km.calinski_harabasz_score_())  # Maximize
            davies_bouldin.append(km.davies_bouldin_score_())  # Minimize
            if verbose:
                print(f'Number of groups: {i} --- Inertia: {ine[i - 2]}')
                print(f'Number of groups: {i} --- Silhouette coefficient: {sil[i - 2]}')
                print(f'Number of groups: {i} --- Calinski_harabasz: {calinski_harabasz[i - 2]}')
                print(f'Number of groups: {i} --- Davies_bouldin: {davies_bouldin[i - 2]}\n')

        # normalizing the scores
        if normalize:
            ine = normalize_(ine)
            sil = normalize_(sil)
            calinski_harabasz = normalize_(calinski_harabasz)
            davies_bouldin = normalize_(davies_bouldin)

        # Index of optimal metric value
        kl = KneeLocator(n_range, ine, curve="convex", direction='decreasing', S=2)

        ylabels = ["Inertia", "Silhouette coefficient", "Calinski harabasz score", "Davies bouldin score"]
        scores = [ine, sil, calinski_harabasz, davies_bouldin]
        indexes = [kl.knee, max_index(sil), max_index(calinski_harabasz), max_index(davies_bouldin)]

        # Plotting
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(26, 13))
        plt.setp(ax[1], xlabel='Number of groups')

        for i in range(4):
            ax[i // 2][i % 2].plot(n_range, scores[i])
            ax[i // 2][i % 2].set_ylabel(ylabels[i])
            if (indexes[i] != None):
                ax[i // 2][i % 2].axvline(x=n_range[indexes[i]], ymax=scores[i].max(), color='red', linestyle='--')
            else:
                print("No knee/elbow found")

        [ax[i // 2][i % 2].set_xticks(n_range) for i in range(0, 2)]
        plt.tight_layout()
        plt.show()

    def create_model(self, n_clusters):
        if(self.use_cuda):
            self.model = cu_KMeans(n_clusters=n_clusters, max_iter=5000, random_state=42)
        else:
            self.model = KMeans(n_clusters=n_clusters, max_iter=5000, random_state=42)
        self.model.fit(self.X)
        return

    def silhouette_plot(self):
        if not self.model:
            print("Create a clustering model first")

        else:
            silhouette_visualizer(self.model, self.X)

        return

    def inertia(self):

        if not self.model:
            print("Create a clustering model first.")
        else:
            return self.model.inertia_


class dbscans(_clustering):
    def __init__(self, data, use_cuda = False):
        super().__init__(data)
        if(use_cuda):
            if(has_rapids_env):
                self.use_cuda = use_cuda
            else:
                self.use_cuda = False;
                print("No rapids enviroments found.\nMoving to default DBScans instead.")
        else:
            self.use_cuda = False;

    def get_baseline_eps(self, n_neighbors=3):

        dists,inds = NNeighbors(X = self.X, n_neighbors=n_neighbors, use_cuda=self.use_cuda)

        # Sort in ascending order
        dists = cp.sort(dists, axis=0) if(has_rapids_env) else np.sort(dists, axis=0)

        # Get the distance of the k'th farthest point
        dists = dists[:, -1]

        n_range = np.arange(0, len(dists), 1) if(has_rapids_env) else cp.arange(0, len(dists), 1)
        kl = KneeLocator(n_range, dists, curve="convex", direction='increasing', S=3)

        knee = kl.knee

        if (not knee):
            print("No knee/elbow found")

        return knee, dists

    def plot_baseline_eps(self, n_neighbors=3):

        knee, dists = self.get_baseline_eps(n_neighbors)
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(dists)

        if (knee != None):
            ax.axhline(xmin=0, xmax=len(dists), y=dists[knee], color='red', linestyle='--')

        ax.set_xlabel("Ascending data index")
        ax.set_ylabel("Minimum intersample distance")
        ax.set_title('Baseline eps value for ' + str(n_neighbors) + ' nearest neighbors')

        ax.legend(['eps values', 'threshold: ' + str(round(dists[knee], 4))])
        plt.show()

        return

    def plot_hyperparameters(self, eps_list = [0.1, 0.15,0.2,0.25,0.3], min_samples = [5,10,15,20,25], figsize=(22, 22), remove_outliers=False):
        
        columns = self.X.columns

        df_size = len(columns)

        if(df_size > 3):
            print("This method is only avaliable for 1,2, and 3 dimensional datasets.")
            return
        
        nrows = len(eps_list);
        ncols = len(min_samples)

        if(df_size != 3):
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, tight_layout=True, sharey=True, sharex=True)
            ax = ax.reshape((nrows * ncols,))
            index = 0


        else:
            fig = plt.figure(figsize=figsize)
            index = 1


        for eps in tqdm(eps_list):

            for min_sample in min_samples:
                
                if(self.use_cuda):
                    dbscan = cu_DBSCAN(eps=eps, min_samples=min_sample).fit(self.X)
                else:
                    dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(self.X)

                labels = dbscan.labels_

                cluster_data = self.X.copy()
                cluster_data['cluster_dbscan'] = labels

                if remove_outliers:
                    cluster_data = remove_outliers_(cluster_data, 'cluster_dbscan')

                
                if(df_size == 1):
                    x_axis = list(range(len(cluster_data)))
                    sns.scatterplot(data=cluster_data, x = x_axis, y = columns[0], hue='cluster_dbscan', ax=ax[index], palette='Set1',
                    legend=False)
                elif(df_size == 2):
                    sns.scatterplot(data=cluster_data, x = columns[0], y = columns[1], hue='cluster_dbscan', ax=ax[index], palette='Set1',
                    legend=False)
                
                else:
                    ax = fig.add_subplot(nrows, ncols, index, projection='3d')
                    ax.scatter(cluster_data[columns[0]],cluster_data[columns[1]],cluster_data[columns[1]], c = labels)
                    ax.set_title('eps = ' + str(eps) + ', min sample = ' + str(min_sample))
                    
                if(df_size!=3):
                    ax[index].set_title('eps = ' + str(eps) + ', min sample = ' + str(min_sample))
                
                index += 1

        plt.show()
        return

    def clustering(self, remove_outliers=False):

        if not self.model:
            print("Create a clustering model first")

        else:
            self.labels = self.model.labels_
            cluster_data = self.X.copy()
            cluster_data['cluster_dbscan'] = self.labels
            if (remove_outliers):
                cluster_data = remove_outliers_(cluster_data, 'cluster_dbscan', -1)
            return cluster_data

    def create_model(self, eps, msamples=5):

        if(self.use_cuda):
            self.model = cu_DBSCAN(eps=eps, min_samples=msamples)
        else:
            self.model = DBSCAN(eps=eps, min_samples=msamples)
        
        self.model.fit(self.X)

        return
class GMM(_clustering):

    def __init__(self, data):
        super().__init__(data)

    cv_types = ['spherical', 'tied', 'diag', 'full']

    
    """def make_ellipses(self, ax, color_pallete="bright"):

        colors = sns.color_palette(color_pallete)
        for n in range(len(self.model.means_)):
            if self.model.covariance_type == "full":
                covariances = self.model.covariances_[n][:2, :2]
            elif self.model.covariance_type == "tied":
                covariances = self.model.covariances_[:2, :2]
            elif self.model.covariance_type == "diag":
                covariances = np.diag(self.model.covariances_[n][:2])
            elif self.model.covariance_type == "spherical":
                covariances = np.eye(self.model.means_.shape[1]) * self.model.covariances_[n]

            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])

            angle = 180 * angle / np.pi  # convert to degrees

            aux = v
            diags = [1.2, 1.7, 2.2]
            for i in diags:
                v = i * np.sqrt(2.0) * np.sqrt(aux)
                ell = mpl.patches.Ellipse(self.model.means_[n, :2], v[0], v[1], 180 + angle, edgecolor=colors[n],
                                          facecolor="none")
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(1)
                ax.add_artist(ell)
                ax.set_aspect("equal", "datalim")"""

    def __plot_scores(self, n_clusters, sil, calinski_harabasz, davies_bouldin, bic):

        n_range = range(2, n_clusters + 1)
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(18, 10))

        plt.setp(ax[1], xlabel='Number of groups')
        [plt.setp(ax[i][0], ylabel="Score") for i in range(0, 2)]
        plt.setp(ax[0][0], title="Silhouette")
        plt.setp(ax[0][1], title="Calinski Harabasz")
        plt.setp(ax[1][0], title="Davies Bouldin")
        plt.setp(ax[1][1], title="BIC")

        sil_index = max_index(sil) % (n_clusters - 1)
        cal_index = max_index(calinski_harabasz) % (n_clusters - 1)
        dav_index = min_index(davies_bouldin) % (n_clusters - 1)
        bic_index = min_index(bic) % (n_clusters - 1)

        ax[0][0].axvline(x=n_range[sil_index], ymax=sil.max(), color='red', linestyle='--', label="optimal")
        ax[0][1].axvline(x=n_range[cal_index], ymax=calinski_harabasz.max(), color='red', linestyle='--',
                         label="optimal")
        ax[1][0].axvline(x=n_range[dav_index], ymax=davies_bouldin.max(), color='red', linestyle='--', label="optimal")
        ax[1][1].axvline(x=n_range[bic_index], ymax=bic.max(), color='red', linestyle='--', label="optimal")

        for i in range(4):
            sil_sub = [sil[(n_clusters - 1) * i + k] for k in range(0, n_clusters - 1)]
            cal_sub = [calinski_harabasz[(n_clusters - 1) * i + k] for k in range(0, n_clusters - 1)]
            dav_sub = [davies_bouldin[(n_clusters - 1) * i + k] for k in range(0, n_clusters - 1)]
            bic_sub = [bic[(n_clusters - 1) * i + k] for k in range(0, n_clusters - 1)]

            ax[0][0].plot(n_range, sil_sub, label=self.cv_types[i])
            ax[0][1].plot(n_range, cal_sub, label=self.cv_types[i])
            ax[1][0].plot(n_range, dav_sub, label=self.cv_types[i])
            ax[1][1].plot(n_range, bic_sub, label=self.cv_types[i])

        [[ax[i][j].set_xticks(n_range) and ax[i][j].legend() for j in range(0, 2)] for i in range(0, 2)]
        plt.tight_layout()
        plt.show()

    def criterion_analysis(self, n_clusters=2, verbose=False, normalize=True):
        """Calculates and displays the silhouette, calinski harabasz and davies boudin scores for each covariance type and 2...n_clusters range.

        Args:
            n_clusters (int): The number of mixture components. Defaults to 2.
            verbose (bool, optional): Enable verbose output. If 1 then it prints the current initialization and each iteration step. 
            normalize (bool, optional): if true then the scores are shown normalized. Defaults to True.
        """

        n_range = range(2, n_clusters + 1)
        self.n_range = n_range

        # intialize the score arrays
        
        sil = cp.zeros(4 * (n_clusters - 1)) if(has_rapids_env) else np.zeros(4 * (n_clusters - 1))
        calinski_harabasz = cp.zeros(4 * (n_clusters - 1)) if(has_rapids_env) else np.zeros(4 * (n_clusters - 1))
        davies_bouldin = cp.zeros(4 * (n_clusters - 1)) if(has_rapids_env) else np.zeros(4 * (n_clusters - 1))
        bic = cp.zeros(4 * (n_clusters - 1)) if(has_rapids_env) else np.zeros(4 * (n_clusters - 1))

        it = 0

        # loops between each covariance type
        for cv_type in tqdm(self.cv_types):

            # creates a model for each cluster group using the current covariance type
            for i in n_range:

                gmm = GaussianMixture(n_components=i, covariance_type=cv_type, max_iter=5000, random_state=42)
                gmm.fit(self.X)

                # saves the scores
                sil[it] = silhouette_score(self.X, gmm.predict(self.X))  # Maximize
                calinski_harabasz[it] = calinski_harabasz_score(self.X, gmm.predict(self.X))  # Maximize
                bic[it] = gmm.bic(self.X)  # Minimize

                if verbose:
                    print(
                        f'Tipo da covariância: {cv_type} - Número de agrupamentos: {i}\n    BIC: {bic[it]}\n    Silhouette: {sil[it]}\n    Davies Bouldin: {davies_bouldin[it]}\n    Calinski Harabasz: {calinski_harabasz[it]}\n')
                it += 1

        # normalizing scores
        if normalize:
            sil = normalize_(sil)
            calinski_harabasz = normalize_(calinski_harabasz)
            davies_bouldin = normalize_(davies_bouldin)
            bic = normalize_(bic)

        # plot the scores
        self.__plot_scores(n_clusters, sil, calinski_harabasz, davies_bouldin, bic)

    def create_model(self, cv_type, n_clusters=2):
        """Create a Gaussian Mixture Model  fitting its class data

        Args:
            cv_type (String): Describes the type of covariance parameters to use. Must be one of:
                'full'
                each component has its own general covariance matrix

                'tied'
                all components share the same general covariance matrix

                'diag'
                each component has its own diagonal covariance matrix

                'spherical'
                each component has its own single variance

            n_clusters (int): The number of mixture components. Defaults to 2.
        """

        # create the model using the given parameters
        self.model = GaussianMixture(n_components=n_clusters, covariance_type=cv_type, max_iter=5000, random_state=42)
        # fit the model using its data
        self.model.fit(self.X)

        return

    def clustering(self, X, prob_threshold=0):
        """Classify the labels for the data samples in X using trained model, and detect outliers using the prob_threshold value.

        Args:
            X (array-like of shape (n_samples, n_features)): List of n_features-dimensional data points. Each row corresponds to a single data point. prob_threshold (int, optional): threshold for interpolation region or outlier detection. If the score of a given point is below the threshold, it is classified as a outlier. Defaults to 0.

        Returns:
            labels (n_samples, n_features): Component labels. -1 for outliers.
        """
        if not self.model:
            print("Create a clustering model first")

        else:
            # predicts the given data
            cluster_data = X.copy()
            cluster_data['cluster'] = self.model.predict(X)

            # creates a array with scores of each prediction
            k = cp.array(self.model.score_samples(X)) if(has_rapids_env) else np.array(self.model.score_samples(X))
            # normalize the array betweeen 0 and 1
            k = (k - k.min()) / (k.max() - k.min())

            # creates a new column with the normalized scores
            cluster_data["scores"] = k

            # set a new cluster group with the possible anomalies in the dataset
            cluster_data.loc[cluster_data['scores'] < prob_threshold, 'cluster'] = -1
            self.labels = cluster_data['cluster']

        # returns the labeled data
        return cluster_data

    def BIC(self):
        if not self.model:
            print("Create a clustering model first.")

        else:
            return self.model.bic(self.X)