import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

from sklearn.manifold import TSNE

import cv2

class pca():
    def __init__(self, df, threshold, label):
        """Object for dimensionality reduction using Principal Component Analysis (PCA)

        Args:
            df (pandas): pandas dataframe containing the dataset
            threshold (float): Variable representing the amount of information we wish to maintain after dimensionality reduction. Between 0 and 1
            label (str): pandas dataframe column name of a label variable
        """
        self.df = df
        self.df_num = self.df._get_numeric_data()
        
        if label:
            self.label_col= self.df[label]
        
        self.threshold = threshold
        
        # Apply PCA
        self.model = PCA()
        self.model.fit_transform(self.df_num)
        self.cumsum = np.cumsum(self.model.explained_variance_ratio_)
        
        # Obtain number of PC to satisfy the threshold criteria
        self.d = np.argmax(self.cumsum >= self.threshold)
        self.num_pc = self.d + 1
        
        self.x_values = np.arange(1, len(self.cumsum)+1, 1)
        
        # Train a PCA model that actually satisfies the threshold criteria
        self.model_red = PCA(n_components=self.num_pc)
        self.model_red.fit_transform(self.df_num)
        self.cumsum_red = np.cumsum(self.model_red.explained_variance_ratio_)
        
    def plot_pca(self):
        """Plot the explained cumulative variance of the dataset to help visualize the information lost during PC pruning.
        """
        plt.scatter(self.x_values, self.cumsum * 100, label='Cumulative')
        plt.scatter(self.x_values, self.model.explained_variance_ratio_ * 100, label='Individual')
        plt.xlabel('Principal component number')
        plt.ylabel('Explained cumulative variance (%)')

        plt.vlines(x=self.num_pc, ymin=0, ymax=100,
                   label='{0} principal components with {1}% explained variance'.format(self.num_pc, 100*np.round(self.cumsum[self.d], 4)),
                   linestyles='dashed',
                   color='red')

        plt.legend()
        plt.show()
        
    def plot_1d(self, label=True, **kwargs):
        """1D visualization of the principal components

        Args:
            label (bool, optional): Color the plot based on the label columns. Defaults to True.

        Returns:
            [figure]: Figure of the plot
        """
            
        PCA2d = PCA(n_components=1, random_state=42, **kwargs)
        plot_data = PCA2d.fit_transform(self.df_num)
        plot_data = pd.DataFrame(plot_data, columns=['PC1'])
        
        cumsum = np.cumsum(PCA2d.explained_variance_ratio_)
        
        if label:
            plot_data['Label'] = self.label_col
            fig = px.scatter(plot_data, y="PC1",
                            color="Label", symbol='Label',
                            width=1200, height=600,
                            title='1D PCA with {}% explained variance'.format(np.round(cumsum[-1]*100, 2)))
        else:
            fig = px.scatter(plot_data, y="PC1",
                            width=1200, height=600,
                            title='1D PCA with {}% explained variance'.format(np.round(cumsum[-1]*100, 2)))

        fig.update_traces(marker=dict(size=4))
        return fig
        
    def plot_2d(self, label=True, **kwargs):
        """2D visualization of the principal components

        Args:
            label (bool, optional): Color the plot based on the label columns. Defaults to True.

        Returns:
            [figure]: Figure of the plot
        """
        
        PCA2d = PCA(n_components=2, random_state=42, **kwargs)
        plot_data = PCA2d.fit_transform(self.df_num)
        plot_data = pd.DataFrame(plot_data, columns=['PC1', 'PC2'])
        
        cumsum = np.cumsum(PCA2d.explained_variance_ratio_)
        
        if label:
            plot_data['Label'] = self.label_col
            fig = px.scatter(plot_data, x="PC1", y="PC2",
                            color="Label", symbol='Label',
                            width=1200, height=600,
                            title='2D PCA with {}% explained variance'.format(np.round(cumsum[-1]*100, 2)))
        else:
            fig = px.scatter(plot_data, x="PC1", y="PC2",
                            width=1200, height=600,
                            title='2D PCA with {}% explained variance'.format(np.round(cumsum[-1]*100, 2)))

        fig.update_traces(marker=dict(size=4))
        return fig


    def plot_3d(self, label=True, **kwargs):
        """3D visualization of the principal components

        Args:
            label (bool, optional): Color the plot based on the label columns. Defaults to True.

        Returns:
            [figure]: Figure of the plot
        """
        
        PCA3d = PCA(n_components=3, random_state=42, **kwargs)
        plot_data = PCA3d.fit_transform(self.df_num)
        plot_data = pd.DataFrame(plot_data, columns=['PC1', 'PC2', 'PC3'])
        
        cumsum = np.cumsum(PCA3d.explained_variance_ratio_)
        
        if label:
            plot_data['Label'] = self.label_col
            fig = px.scatter_3d(data_frame=plot_data,
                            x='PC1', y='PC2', z='PC3',
                            color='Label',
                            title='3D PCA with {}% explained variance'.format(np.round(cumsum[-1]*100, 2)),
                            width=1200, height=600)
        else:
            fig = px.scatter_3d(data_frame=plot_data,
                                x='PC1', y='PC2', z='PC3',
                                title='3D PCA with {}% explained variance'.format(np.round(cumsum[-1]*100, 2)),
                                width=1200, height=600)


        fig.update_traces(marker=dict(size=3))
        
        return fig

    def reconstruction_error(self,df_reduced):
        """Visualize reconstruction error when applying the inverse PCA transform on the pruned dataset

        Args:
            df_reduced (pd): Reduced pandas dataframe 

        Returns:
            float: Reconstruction error matrix 
        """
        fig, ax = plt.subplots(1,3,sharey=True,figsize=(15, 5))
        plt.suptitle('Dataset reconstruction with {0} principal components and {1}% explained variance'.format(self.num_pc, round(self.cumsum_red[-1]*100, 2)), y=1.01)

        vmin = df_reduced.min().min()
        vmax = df_reduced.max().max()
        
        im0 = ax[0].imshow(self.df_num.to_numpy(), cmap=plt.get_cmap("seismic"), vmin=vmin, vmax=vmax, aspect='auto')
        fig.colorbar(im0, ax=ax[0], ticks=None)
        ax[0].set_title('Original data')

        print('Original data plotted')
        
        im1 = ax[1].imshow(df_reduced, cmap=plt.get_cmap("seismic"), vmin=vmin, vmax=vmax, aspect='auto')
        fig.colorbar(im1, ax=ax[1], ticks=None)
        ax[1].set_title('Reduced data')

        print('Reduced data plotted')
        
        diff = abs(self.df.to_numpy() - df_reduced)
        im2 = ax[2].imshow(diff, cmap=plt.get_cmap("seismic"), aspect='auto')
        fig.colorbar(im2, ax=ax[2], ticks=None)
        ax[2].set_title('Absolute error')
        plt.tight_layout()
        
        print('Error plotted')
        
        return diff #fig, ax, diff
    
    
    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))
        return

    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))
        return



#########################
#  TSNE
#########################

class tsne_plot():
    def __init__(self, data):
        
        self.df = data.copy()
        self.df_type = df_type

    def plot2D(self, **kwargs):

        tsne = TSNE(n_components=2, random_state=42, **kwargs)
        plot_data = tsne.fit_transform(self.df)
        plot_data = pd.DataFrame(plot_data, columns=['C1', 'C2'])

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_xlabel('C1', size=20)
        ax.set_ylabel('C2', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        sns.scatterplot(data=plot_data, x='C1', y='C2', ax=ax)

        plt.show()

    def plot3D(self, **kwargs):

        tsne = TSNE(n_components=3, random_state=42, **kwargs)
        plot_data = tsne.fit_transform(self.df)
        plot_data = pd.DataFrame(plot_data, columns=['C1', 'C2', 'C3'])

        fig = px.scatter_3d(data_frame=plot_data,
                                x='C1', y='C2', z='C3',
                                width=800, height=800)

        fig.show()
        return
