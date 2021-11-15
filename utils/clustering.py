import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
from yellowbrick.cluster import silhouette_visualizer
from sklearn.neighbors import NearestNeighbors

class _clustering():

    def __init__(self, df, cat=False):
        self.X = df.copy()
        self.model = None


    def clustering(self, X):
        """Predict the labels for the data samples in X using trained model.



        Args:
            X (array-like of shape (n_samples, n_features)): List of n_features-dimensional data points. Each row corresponds to a single data point.

        Returns:
            labels (array, shape (n_samples,): Component labels.
        """
        if not self.model:
            print("Create a clustering model first")

        else:
            self.Y_ = self.model.predict(X)
            cluster_data = self.X.copy()
            cluster_data['cluster'] = self.Y_

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



class kmeans(_clustering):

    def __init__(self,data):
        super().__init__(data)
    
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
        
        else:
            silhouette_visualizer(self.model, self.X)
        
        return

class dbscans(_clustering):
    def __init__(self,data):
        super().__init__(data)
    
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

    def clustering(self):

        if not self.model:
            print("Create a clustering model first")

        else:
            self.Y_ = self.model.labels_
            cluster_data = self.X.copy()
            cluster_data['cluster_dbscan'] = self.Y_
            return cluster_data

    def create_model(self, eps, msamples=5):
        self.model = DBSCAN(eps=eps, min_samples=msamples, n_jobs=-1)
        self.model.fit(self.X)
        return



class GMM(_clustering):
    
    def __init__(self,data):
        super().__init__(data)

    cv_types = ['spherical', 'tied', 'diag', 'full']
    

    def __plot_scores(self, n_clusters, sil, calinski_harabasz, davies_bouldin, bic, normalize = True):

        n_range = range(2, n_clusters + 1)
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(18, 10))
        sil_y_max = sil.max()
        cal_y_max = calinski_harabasz.max()
        dav_y_max = davies_bouldin.max()
        bic_y_max = bic.max()
        
        
        yticks = [i/4 for i in range(5)]

        plt.setp(ax[1], xlabel='Number of groups')
        [plt.setp(ax[i][0], ylabel = "Score") for i in range(0,2)]
        plt.setp(ax[0][0],title = "Silhouette")
        plt.setp(ax[0][1],title = "Calinski Harabasz")
        plt.setp(ax[1][0],title = "Davies Boudini")
        plt.setp(ax[1][1],title = "BIC")


        sil_index = np.where(sil == np.amax(sil))[0][0]%(n_clusters-1)
        cal_index = np.where(calinski_harabasz == np.amax(calinski_harabasz))[0][0]%(n_clusters-1)
        dav_index = np.where(davies_bouldin == np.amin(davies_bouldin))[0][0]%(n_clusters-1)
        bic_index = np.where(bic == np.amin(bic))[0][0]%(n_clusters-1)


        ax[0][0].axvline(x=n_range[sil_index],ymax = sil_y_max, color='red', linestyle='--', label = "optimal")
        ax[0][1].axvline(x=n_range[cal_index], ymax=cal_y_max, color='red', linestyle='--', label = "optimal")
        ax[1][0].axvline(x=n_range[dav_index],ymax=dav_y_max, color='red', linestyle='--', label = "optimal")
        ax[1][1].axvline(x=n_range[bic_index],ymax=bic_y_max, color='red', linestyle='--', label = "optimal")
        
        for i in range(4):
                
            sil_sub = [sil[(n_clusters-1)*i + k] for k in range(0,n_clusters-1)]
            cal_sub = [calinski_harabasz[(n_clusters-1)*i + k] for k in range(0,n_clusters-1)]
            dav_sub = [davies_bouldin[(n_clusters-1)*i + k] for k in range(0,n_clusters-1)]
            bic_sub = [bic[(n_clusters-1)*i + k] for k in range(0,n_clusters-1)]

            ax[0][0].plot(n_range, sil_sub, label=self.cv_types[i])
            ax[0][1].plot(n_range, cal_sub, label=self.cv_types[i])
            ax[1][0].plot(n_range, dav_sub, label=self.cv_types[i])
            ax[1][1].plot(n_range, bic_sub, label=self.cv_types[i])
            
        [[ax[i][j].set_xticks(n_range) and ax[i][j].legend() for j in range(0,2)] for i in range(0,2)]
        plt.tight_layout()  
        plt.show()


    def criterion_analysis(self, n_clusters=2, verbose=False, normalize = True):
        """Calculates and displays the silhouette, calinski harabasz and davies boudin scores for each covariance type and 2...n_clusters range.

        Args:
            n_clusters (int): The number of mixture components. Defaults to 2.
            verbose (bool, optional): Enable verbose output. If 1 then it prints the current initialization and each iteration step. 
            normalize (bool, optional): if true then the scores are shown normalized. Defaults to True.
        """
    
        n_range = range(2, n_clusters + 1)
        self.n_range = n_range

        #intialize the score arrays
        sil = np.zeros(4*(n_clusters-1))
        calinski_harabasz = np.zeros(4*(n_clusters-1))
        davies_bouldin = np.zeros(4*(n_clusters-1))
        bic = np.zeros(4*(n_clusters-1))
        it = 0
        
        #loops between each covariance type
        for cv_type in tqdm(self.cv_types):

            #creates a model for each cluster group using the current covariance type
            for i in n_range:
                
                gmm = GaussianMixture(n_components=i, covariance_type=cv_type, max_iter=5000, random_state=42)
                gmm.fit(self.X)
                
                #saves the scores
                sil[it] = silhouette_score(self.X, gmm.predict(self.X))                             # Maximize
                calinski_harabasz[it] = calinski_harabasz_score(self.X, gmm.predict(self.X))        # Maximize
                davies_bouldin[it] = davies_bouldin_score(self.X, gmm.predict(self.X))              # Minimize
                bic[it] = gmm.bic(self.X)                                                           # Minimize
                

                if verbose:
                    print(f'Tipo da covariância: {cv_type} - Número de agrupamentos: {i}\n    BIC: {bic[it]}\n    Silhouette: {sil[it]}\n    Davies Bouldin: {davies_bouldin[it]}\n    Calinski Harabasz: {calinski_harabasz[it]}\n')
                it+=1

        #normalizing the scores
        if normalize:
            sil = (sil-sil.min()) / (sil.max()-sil.min())
            calinski_harabasz = (calinski_harabasz-calinski_harabasz.min())/(calinski_harabasz.max()-calinski_harabasz.min())
            davies_bouldin = (davies_bouldin-davies_bouldin.min())/(davies_bouldin.max()-davies_bouldin.min())
            bic = (bic-bic.min())/(bic.max()-bic.min())


        #plot the scores
        self.__plot_scores(n_clusters, sil, calinski_harabasz, davies_bouldin, bic, normalize=normalize)
        
    
    def create_model(self, cv_type, n_clusters= 2):
        """Create a Gaussian Mixture Model  fitting its class data

        Args:
            cv_type (String): Describes the type of covariance parameters to use. Must be one of:
                ‘full’
                each component has its own general covariance matrix

                ‘tied’
                all components share the same general covariance matrix

                ‘diag’
                each component has its own diagonal covariance matrix

                ‘spherical’
                each component has its own single variance

            n_clusters (int): The number of mixture components. Defaults to 2.
        """

        #create the model using the given parameters
        self.model = GaussianMixture(n_components=n_clusters, covariance_type=cv_type, max_iter=5000, random_state=42)
        #fit the model using its data
        self.model.fit(self.X)

        return


    def clustering(self, X, prob_threshold = 0):
        """Classify the labels for the data samples in X using trained model, and detect outliers using the prob_threshold value.

        Args:
            X (array-like of shape (n_samples, n_features)): List of n_features-dimensional data points. Each row corresponds to a single data point. prob_threshold (int, optional): threshold for interpolation region or outlier detection. If the score of a given point is below the threshold, it is classified as a outlier. Defaults to 0.

        Returns:
            labels (n_samples, n_features): Component labels. -1 for outliers.
        """
        if not self.model:
            print("Create a clustering model first")

        else:
            #predicts the give data
            self.Y_ = self.model.predict(X)
            cluster_data = X.copy()
            cluster_data['cluster'] = self.Y_


            #creates a array with scores of each prediction
            k = np.array(self.model.score_samples(X))
            #normalize the array betweeen 0 and 1
            k = (k-k.min()) / (k.max()-k.min())

            #creates a new column with the normalized scores
            cluster_data["scores"] = k

            #set a new cluster group with the possible anomalies in the dataset
            cluster_data.loc[cluster_data['scores'] < prob_threshold, 'cluster'] = -1
            
        #returns the labeled data
        return cluster_data