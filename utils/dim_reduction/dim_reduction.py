from sklearn.decomposition import PCA as SKLPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from optht import optht


#########################
#  PCA
#########################

class PCA():
    """
    Compiles the defaut steps for a PCA usage, adding different methods for dimensional
    reduction and graph plottings for visualization and analyzes.
    """

    def __init__(self, x, strategy='oht', n_components=.9):
        """
        Validates the input data, normalizes it for variance 1 and mean 0, creates the
        PCA model and calculates the cutoff based on the choosen dimensional reduction
        method.

        Args:
            x (numpy.ndarray): 2-D array containing the numeric values of the dataset
            to be processed, validated for dimensionality and existence of both NaN 
            and infinite values in the '_validate_dataset' method.

            strategy (string): Strategy to be utilized for dimensional cut.

            n_components (float): The amount of components to be kept in PCA. If the value
            is in the (0, 1) range, the value is going to represent the amount of cumulative
            energy (sum of singular values ratios) to be kept. If the it's a integer > 0, it
            will represent the absolute quantity of singular values to be kept.
        """
        self._validate(x, strategy, n_components)

        self.x = x
        self.x_normalized = StandardScaler().fit_transform(x)

        self.strategy = strategy
        self.n_components = n_components

        self.pca = SKLPCA()
        self.pca.fit_transform(self.x_normalized)

        if strategy == 'oht':
            self.concrete_strategy = OHTStrategy()
            self.pca_cut = self.concrete_strategy.get_cut_pca(self.x_normalized, self._get_singular_values())
        elif strategy == 'arbitrary cut':
            self.concrete_strategy = ArbitraryCutStrategy()
            self.pca_cut = self.concrete_strategy.get_cut_pca(self.n_components)
        else:
            raise ValueError('Please choose a valid strategy')

        self.x_cut = self.pca_cut.fit_transform(self.x_normalized)


    def _validate(self, x, strategy, n_components): 
        """
        Validates certain conditions for PCA and the class to work.

        Args:
            x (numpy.ndarray): Dataset matrix.

            oht (boolean): If the class is going to use the OHT for a dimensional cut.

            n_components: The percentage of components to be kept in PCA.

        Raises:
            ValueError
        """
        if x.ndim != 2:
            raise ValueError('The dataset needs to have 2 dimensions')
        if np.isnan(x).any():
            raise ValueError('There are NaN values in the dataset')
        if not np.isfinite(x).all():
            raise ValueError('There are infinite values in the dataset')


    def _get_singular_values(self):
        """
        Returns:
            numpy.ndarray: PCA's singular values.
        """
        return self.pca.singular_values_


    def _get_singular_value_ratio_cumsum(self):
        """
        Returns:
            numpy.ndarray: PCA's cumulative singular value ratios.
        """
        svs = self._get_singular_values()
        return np.cumsum(svs / sum(svs))

    def get_sv_index_cut(self):
        return self.pca_cut.n_components

    
    def get_cut_dataset(self):
        """
        Returns:
            [numpy.ndarray]: Dimensionally cut data.
        """
        return self.x_cut

    
    def plot_cumulative_energy(self, log=False, arbitrary_threshold=.9):
        """
        Plots the cumulative singular value raio and draws two cutoff lines: one abritary
        and another for the Optimal Hard Threshold.
        """
        cumsum = self._get_singular_value_ratio_cumsum()
        cumsum_df = pd.DataFrame({
            'Singular value index': range(len(cumsum)), 
            'Cumulative energy': cumsum
        })

        fig = px.line(
            cumsum_df, 
            y='Cumulative energy', 
            x='Singular value index', 
            title="Cumulative energy of the PCA's singular values", 
            markers=True,
            log_y = log
        )

        if self.strategy == 'oht':
            fig.add_hline(
                y=cumsum[self.get_sv_index_cut()], 
                line_dash='dash', 
                line_color='orange', 
                annotation_text='Optimal Hard Threshold', 
                annotation_position='top left'
            )
            fig.add_hline(
                y=arbitrary_threshold, 
                line_dash='dash', 
                line_color='red', 
                annotation_text=f'Arbitrary {(arbitrary_threshold * 100):.1f}% Threshold', 
                annotation_position='bottom left'
            )
        elif self.strategy == 'arbitrary cut':
            fig.add_hline(
                y=self.n_components, 
                line_dash='dash', 
                line_color='red', 
                annotation_text=f'Arbitrary {(self.get_sv_index_cut() * 100):.1f}% Threshold', 
                annotation_position='bottom left'
            )
        
        fig.show()


    def plot_singular_values(self, log=True, arbitrary_threshold=.9):
        svs = self._get_singular_values()
        svs_df = pd.DataFrame({
            'Singular value index': range(len(svs)), 
            'Singular value': svs
        })

        cumulative = 0
        threshold = sum(svs) * arbitrary_threshold
        arbitrary_cut = 0
        # this tries to get an approximation of where the threshold would be placed
        # idk just empirical shit that came to my mind i'm no mathematician
        for i in range(len(svs)):
            cumulative += svs[i]
            if cumulative >= threshold:
                if i == len(svs) - 1:
                    arbitrary_cut = svs[i]
                else:
                    arbitrary_cut = svs[i+1] + (svs[i] - svs[i+1]) * ((svs[i+1] - (cumulative - threshold)) / svs[i+1])
                break

        fig = px.line(
            svs_df, 
            y='Singular value', 
            x='Singular value index', 
            title="PCA's singular values", 
            markers=True,
            log_y = log
        )

        if self.strategy == 'oht':
            fig.add_hline(
                y=svs[self.get_sv_index_cut()], 
                line_dash='dash', 
                line_color='orange', 
                annotation_text='Optimal Hard Threshold'
            )
            fig.add_hline(
                y=arbitrary_cut, 
                line_dash='dash', 
                line_color='red', 
                annotation_text=f'Arbitrary {(arbitrary_threshold * 100):.1f}% Threshold'
            )
        elif self.strategy == 'arbitrary cut':
            fig.add_hline(
                y=arbitrary_cut, 
                line_dash='dash', 
                line_color='red', 
                annotation_text=f'Arbitrary {(self.get_sv_index_cut() * 100):.1f}% Threshold', 
            )
        
        fig.show()


    def plot_2d(self, label=False, **kwargs):
        """
        Visualization of the dataset reduced to two dimensions.

        Args:
            label (bool, optional): [description]. Defaults to False.
        """
        pca_2d = SKLPCA(n_components=2, random_state=42, **kwargs)
        plot_data = pca_2d.fit_transform(self.x_normalized)
        plot_df = pd.DataFrame(plot_data, columns=['PC1', 'PC2'])

        svs = pca_2d.singular_values_
        cumsum = np.cumsum(svs / sum(svs))

        if label:
            plot_df['Label'] = self.label_col
            fig = px.scatter(plot_df, x="PC1", y="PC2",
                             color="Label", symbol='Label',
                             width=1200, height=600,
                             title='2D PCA with {}% explained variance'.format(np.round(
                                 self._get_singular_value_ratio_cumsum()[1]*100, 2)))
        else:
            fig = px.scatter(plot_df, x="PC1", y="PC2",
                             width=1200, height=600,
                             title='2D PCA with {}% explained variance'.format(np.round(
                                 self._get_singular_value_ratio_cumsum()[1]*100, 2)))

        fig.update_traces(marker=dict(size=4))
        fig.show()


    def plot_3d(self, label=False, **kwargs):
        """
        Visualization of the dataset reduced to three dimensions.

        Args:
            label (bool, optional): [description]. Defaults to False.
        """
        pca_3d = SKLPCA(n_components=3, random_state=42, **kwargs)
        plot_data = pca_3d.fit_transform(self.x_normalized)
        plot_df = pd.DataFrame(plot_data, columns=['PC1', 'PC2', 'PC3'])

        svs = pca_3d.singular_values_
        cumsum = np.cumsum(svs / sum(svs))

        if label:
            plot_df['Label'] = self.label_col
            fig = px.scatter_3d(data_frame=plot_df,
                                x='PC1', y='PC2', z='PC3',
                                color='Label',
                                title='3D PCA with {}% explained variance'.format(
                                    np.round(self._get_singular_value_ratio_cumsum()[2]*100, 2)),
                                width=1200, height=600)
        else:
            fig = px.scatter_3d(data_frame=plot_df,
                                x='PC1', y='PC2', z='PC3',
                                title='3D PCA with {}% explained variance'.format(
                                    np.round(self._get_singular_value_ratio_cumsum()[2]*100, 2)),
                                width=1200, height=600)

        fig.update_traces(marker=dict(size=3))
        fig.show()


class OHTStrategy:

    def get_cut_pca(self, x_normalized, singular_values):
        sv_index_cut, _ = optht(x_normalized, singular_values)
        return SKLPCA(n_components=sv_index_cut)


class ArbitraryCutStrategy:

    def get_cut_pca(self, n_components):
        return SKLPCA(n_components=n_components)


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
