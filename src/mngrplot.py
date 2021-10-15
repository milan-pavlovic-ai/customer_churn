
import time as t
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, roc_curve


class PlotManager:
    """
    Plot Manager
    """

    CLASS_LABEL = 'status_num'

    def __init__(self):
        super().__init__()
        return

    @staticmethod
    def plot_distrib_classes(dataset):
        """
        Plot distribution of classes
        """
        # Distribution
        class_label = PlotManager.CLASS_LABEL
        distrib = dataset[class_label].value_counts().sort_index()
        print('Distribution:', distrib, sep='\n')

        # Plot
        plt.figure()
        x = distrib.index
        y = distrib.values
        rank = y.argsort().argsort()
        color_palette = sns.color_palette('coolwarm', n_colors=len(x))
        palette = np.array(color_palette)[rank]
        axes = sns.barplot(x=x, y=y, palette=palette, edgecolor='gray')
        axes.set_xlabel(class_label)
        axes.set_ylabel('Instances')
        plt.show()
        plt.close()
        return

    @staticmethod
    def plot_feature_distrib(dataset):
        """
        Feature distribution
        """
        for col in dataset.columns:
            plt.figure()
            sns.histplot(data=dataset, x=col, bins=100, kde=True)
            plt.figure()
            sns.boxplot(data=dataset, x=col)
            plt.show()
            plt.close()
        return

    @staticmethod
    def corr_feature_target(dataset):
        """
        Return Pearson correlation between features and target value
        Interpretation: The higer correlation means the class id is higer and vice versa 
        Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        """
        class_label = PlotManager.CLASS_LABEL
        y_data = dataset[class_label]
        X_data = dataset.drop(class_label, axis=1)

        features_corrs = []
        num_features = X_data.shape[1]
        features = np.arange(num_features)

        # Calculate correlation for each feature
        for col in range(num_features):
            feature = X_data.iloc[:, col].to_numpy()
            corr, _ = stats.pearsonr(feature, y_data)
            features_corrs.append(corr)

        # Info
        print('Correlations Feature-Target:\n', features_corrs)
        
        # Plot
        rank = np.array(features_corrs).argsort().argsort()
        color_palette = sns.color_palette('coolwarm', n_colors=num_features)
        palette = np.array(color_palette)[rank]
        axes = sns.barplot(x=features, y=features_corrs, palette=palette, edgecolor=None)
        axes.set_xlabel('Feature')
        axes.set_ylabel('Pearson coef to target')
        axes.set_xticklabels(X_data.columns, rotation=90)
        plt.show()
        plt.close()
        return

    @staticmethod
    def plot_activation(dataset):
        """
        Plot activation
        """
        return

    @staticmethod
    def visualize_data(dataset):
        """
        Visualization of given dataset
        """
        # Classes distribution
        PlotManager.plot_distrib_classes(dataset)

        # Feature distribution
        PlotManager.plot_feature_distrib(dataset)

        # Correlation feature-target
        PlotManager.corr_feature_target(dataset)
        return

    @staticmethod
    def feature_importance(features, importances, sort=True, title=None):
        """
        Feature importance as bar plot
        """
        # Info
        print(importances)
        x_features = np.arange(len(features))
        
        # Sort Numeric Features
        if sort:
            sorted_order = features.argsort()
            features = features[sorted_order]
            importances = importances[sorted_order]

        # Plot
        rank = np.array(importances).argsort().argsort()
        color_palette = sns.color_palette('coolwarm', n_colors=len(features))
        palette = np.array(color_palette)[rank]
        axes = sns.barplot(x=x_features, y=importances, palette=palette, edgecolor=None)
        axes.set_xticklabels(features, rotation=90)
        axes.set_title(title, fontsize=18, pad=20)
        axes.set_xlabel('Feature', fontsize=12)
        axes.set_ylabel('Importance', fontsize=12)
        plt.show()
        plt.close()
        return


if __name__ == "__main__":
    mngr = PlotManager()