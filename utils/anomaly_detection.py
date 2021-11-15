import pickle
import matplotlib.pyplot as plt
from utils.plotting import *

from sklearn.ensemble import IsolationForest


class iforest():
    def __init__(self, data):
        self.X = data.copy()
        self.model = None

    #dev only
    def __classify(self, contamination_range = [0.01, 0.03, 0.05, 0.07], bootstrap = False):
        
        nrows = 2
        ncols = 2
        fig1, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[15,10],tight_layout=True, sharey=True, sharex=True)
        ax = ax.reshape((nrows*ncols,))
        index = 0
        for i in tqdm(range(0,nrows*ncols)):
            mod = IsolationForest( max_samples=1.0, n_jobs=-1, bootstrap= bootstrap, contamination=contamination_range[i])
            mod.fit(self.X)
            
            data_ = self.X.copy()
            data_['scores']=mod.decision_function(data_)
            data_['cont= ' + str(contamination_range[i])] = mod.predict(self.X)

            
            sns.scatterplot(data=data_, x='PC1', y='PC2', size = 1, hue = 'cont= ' + str(contamination_range[i]), ax=ax[index], palette='Set1', legend=False)
            ax[index].set_title('cont= ' + str(contamination_range[i]))
            index+=1
    
        plt.show()
        
        return


    def create_model(self, contamination = .01, **kwargs):
        """Calculates and returns the anomaly score of each sample using the IsolationForest algorithm

        Args:
            contamination (float, optional): The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples. Defaults to .01.
        """
        self.model = IsolationForest(max_samples=1.0, n_jobs=-1, contamination=contamination, **kwargs)
        self.model.fit(self.X)
        return

    def predict_anomalies(self, X):
        """Predict if a particular sample is an outlier or not.

        Args:
            X (array-like of shape (n_samples, n_features)): The input samples. Internally, it will be converted to dtype=np.float32.

        Returns:
            labels (n_samples, n_features): For each observation, tells its score and whether or not (True or False) it should be considered as an inlier according to the fitted model.
        """
        if not self.model:
            print("Create a model with the method create_model first")

        else:
            
            new_data = X.copy()
            #predicts the given data
            new_data['classification'] = self.model.predict(X)

            #sets true for valid data and false for anomalies
            new_data.loc[new_data['classification'] == 1, 'classification'] = True
            new_data.loc[new_data['classification'] == -1, 'classification'] = False
            
            #creates a new column with the scores of each point
            new_data['score'] = self.model.decision_function(X)

        return new_data

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
