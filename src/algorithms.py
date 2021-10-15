
# Basic
import numpy as np
import scipy.stats as stats

# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Supervised
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


class Algorithms:
    """
    Machine leraning algorithms
    """
    
    RANDOM_FORSET = 'RandomForest'
    LOGISTIC_REGRESSION = 'LogisticRegression'

    @staticmethod
    def get_alogirthm(algorithm):
        """
        Returns algorithm for given name
        """
        if algorithm == Algorithms.LOGISTIC_REGRESSION:
            return Algorithms.logistic_regression()
        elif algorithm == Algorithms.RANDOM_FORSET:
            return Algorithms.random_forest()
        else:
            raise ValueError('Not implemented')

    @staticmethod
    def logistic_regression():
        """
        Logistic Regression model
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
        """
        # Define workflow
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest()),
            #('pca', PCA()),
            ('estimator', LogisticRegression(multi_class='multinomial', n_jobs=-1, verbose=1))
        ])
        
        # Define Hyperparamter space
        hparams = [{
            'kbest__score_func': [f_classif, mutual_info_classif],         # scoring function for feature selection
            'kbest__k': np.arange(15, 35),                                   # number of feature to select with best score

            #'pca__n_components': np.arange(3, 8),                       # number of principal axes in feature space to keep

            'estimator__solver':['newton-cg', 'saga', 'sag', 'lbfgs'],       # Algorithm to use in the optimization problem
            'estimator__max_iter': np.arange(105, 220),                         # Maximum number of iterations taken for the solvers to converge
            'estimator__penalty': ['l2', 'none'],                            # Used to specify the norm used in the penalization
            'estimator__C': stats.loguniform(1e-5, 150),                           # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
            'estimator__tol': stats.loguniform(1e-7, 1e-5),                  # Tolerance for stopping criteria
            'estimator__class_weight': ['balanced']                     # Weights associated with classes
        }]

        return pipeline, hparams

    @staticmethod
    def random_forest():
        """
        Random Forest
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        # Define workflow
        pipeline = Pipeline([
            ('scaler', StandardScaler()),       # required for PCA
            #('pca', PCA()),
            ('kbest', SelectKBest()),
            ('estimator', RandomForestClassifier(verbose=1, n_jobs=-1, warm_start=False))
        ])
        
        # Define Hyperparamter space
        hparams = [{
            #'pca__n_components': np.arange(25, 35),                             # desired dimensionality of output data

            'kbest__score_func': [f_classif],          # scoring function for feature selection
            'kbest__k': np.arange(25, 32),                                # number of feature to select with best score

            'estimator__n_estimators': np.arange(100, 170),                     # the number of trees in the forest
            'estimator__criterion': ['entropy', 'gini'],                        # the function to measure the quality of a split
            'estimator__max_depth': np.arange(10, 15),                         # the maximum depth of the tree
            #'estimator__min_samples_split': np.arange(3, 70),                 # the minimum number of samples required to split an internal node
            #'estimator__min_samples_leaf': np.arange(2, 20),                  # the minimum number of samples required to be at a leaf node
            'estimator__max_features': ['sqrt', 'log2'],                        # the number of features to consider when looking for the best split
            'estimator__bootstrap': [True],                              # whether bootstrap samples are used when building trees, if False, the whole dataset is used to build each tree
            'estimator__max_samples': stats.uniform(0.5, 0.4),                 # if bootstrap is True, the number of samples to draw from X to train each base estimator
            'estimator__class_weight': ['balanced', 'balanced_subsample']
        }]

        return pipeline, hparams

