import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

from sklearn.manifold import TSNE


class pca():
    def __init__(self, df, threshold, label=None):
        self.df = df
        self.df_num = self.df._get_numeric_data()
        self.label_col= self.df[label]
        
        self.threshold = threshold
        
        
        self.model = PCA()
        self.model.fit_transform(self.df_num)
        self.cumsum = np.cumsum(self.model.explained_variance_ratio_)
        
        self.d = np.argmax(self.cumsum >= self.threshold)
        self.num_pc = self.d + 1
        
        self.x_values = np.arange(1, len(self.cumsum)+1, 1)
        
        self.model_red = PCA(n_components=self.num_pc)

    def plot_pca(self):
        
        plt.scatter(self.x_values, self.cumsum * 100, label='Cumulative')
        plt.scatter(self.x_values, self.model.explained_variance_ratio_ * 100, label='Individual')
        plt.xlabel('# Principal component')
        plt.ylabel('Explained cumulative variance (%)')

        plt.vlines(x=self.num_pc, ymin=0, ymax=100,
                   label='{0} principal components with {1}% explained variance'.format(self.num_pc, 100*np.round(self.cumsum[self.d], 4)),
                   linestyles='dashed',
                   color='red')

        plt.legend()
        plt.show()
        
    def plot_2d(self, label=True, **kwargs):
        
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

        #fig.show()
        fig.update_traces(marker=dict(size=4))
        return fig


    def plot_3d(self, label=True, **kwargs):
        
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
    def __init__(self, data, df_type=False, sample=False):
        '''
        Um objeto que plota o tsne de dados quaisquer.
        I/O:
            data: um pandas dataframe contendo os dados a sofrerem redução dimensional;
            df_type: um booleano indicando o nome da coluna que pode ser usada para distinguir grupos dentro do conjunto de dados. Se False, não há grupos;
            sample: um booleano indicando quando se deve fazer uma amostragem estratificada com, no mínimo, 99% de confiança. Só é possível realizar a amostragem caso df_type=True;
        '''
        self.df = data.copy()
        self.df_type = df_type

        if df_type and sample:
            # amostragem estratificada para agilizar o treinamento do modelo
            g = data.groupby(df_type)
            u_groups = data[df_type].unique()
            gs = [g.get_group(i).sample(n=process.n_sample(
                g.get_group(i)), random_state=42) for i in u_groups]
            self.df = pd.concat(gs)

    def plot2D(self, **kwargs):
        '''
        Uma função que realiza o plot 2D do TSNE.
        I/O:
            **kwargs: parâmetros do método TSNE do sklearn.manifold.
        '''
        if self.df_type:
            tsne = TSNE(n_components=2, random_state=42, **kwargs)
            plot_data = tsne.fit_transform(self.df.drop(columns=self.df_type))
            plot_data = pd.DataFrame(plot_data, columns=['C1', 'C2'])
            plot_data[self.df_type] = np.array(self.df.loc[:, self.df_type])
            grouped = plot_data.groupby(self.df_type)

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_xlabel('C1', size=20)
            ax.set_ylabel('C2', size=20)
            plt.xticks(size=20)
            plt.yticks(size=20)
            for group in plot_data[self.df_type].unique():
                ax.scatter(grouped.get_group(group)[
                           'C1'], grouped.get_group(group)['C2'], label=group)
            ax.legend()
        else:
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
        '''
        Uma função que realiza o plot 3D do TSNE.
        I/O:
            **kwargs: parâmetros do método TSNE do sklearn.manifold.
        '''
        if self.df_type:
            tsne = TSNE(n_components=3, random_state=42, **kwargs)
            plot_data = tsne.fit_transform(self.df.drop(columns=self.df_type))
            plot_data = pd.DataFrame(plot_data, columns=['C1', 'C2', 'C3'])
            plot_data[self.df_type] = np.array(self.df.loc[:, self.df_type])

            fig = px.scatter_3d(data_frame=plot_data,
                                x='C1', y='C2', z='C3',
                                color=plot_data[self.df_type].astype('object'),
                                width=800, height=800)
        else:
            tsne = TSNE(n_components=3, random_state=42, **kwargs)
            plot_data = tsne.fit_transform(self.df)
            plot_data = pd.DataFrame(plot_data, columns=['C1', 'C2', 'C3'])

            fig = px.scatter_3d(data_frame=plot_data,
                                x='C1', y='C2', z='C3',
                                width=800, height=800)

        fig.show()
        return
