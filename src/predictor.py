
# Basic
import os
import joblib
import time as t
import numpy as np
import pandas as pd
import scipy.stats as stats

# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

# Metrics and evaluation
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, classification_report, accuracy_score, f1_score, roc_auc_score

# Supervised
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Custom
from mngrdata import DataManager
from mngrplot import PlotManager


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


class PredictorManager:
    """
    Predictor manager
    """

    MODEL_DIR = 'data/output'

    def __init__(self, test_size=0.1, info=False):
        """
        Constructor
        """
        super().__init__()
        self.mngrdata = DataManager()
        self.mngrdata.get_dataset(test_size=test_size, info=info)
        self.X_train = self.mngrdata.X_train
        self.y_train = self.mngrdata.y_train
        self.X_test = self.mngrdata.X_test
        self.y_test = self.mngrdata.y_test
        self.dataset = self.mngrdata.dataset
        return

    def random_search_supervised_cv(self, algorithm, pipeline, hparams, n_iters=3, n_folds=3):
        """
        Random search optimization
        """
        # Define metrics
        #   For each trained model with specific combination of hparam run this metrics (in each iteration of CV)
        #   GridSearchCV.cv_results_ will return scoring metrics for each of the score types provided
        metrics = {}
        metrics['f1'] = make_scorer(f1_score, average="macro")
          
        # Define best estimator metric
        #   Refit an estimator on the whole dataset (full training data) using the best found parameters
        #   For multiple metric evaluation, this needs to be a str denoting the scorer/metric that would be used to find the best parameters for refitting the estimator at the end.
        refit_metric = 'f1'

        # Definition of search strategy
        cross_validation = StratifiedKFold(n_splits=n_folds, shuffle=True)
        
        print('Tuning hyperparameters for metrics=', metrics.keys())
        rand_search = RandomizedSearchCV(
            estimator=pipeline, 
            param_distributions=hparams, 
            scoring=metrics,
            n_iter=n_iters,
            refit=refit_metric,
            cv=cross_validation,                                            # for every hparam combination it will shuffle data with same key, random_state
            return_train_score=False,                                       # used to get insights on how different parameter settings impact the overfitting/underfitting trade-off
            n_jobs=-1,                                                      # number of jobs to run in parallel, None means 1 and -1 means using all processors
            verbose=10)                                                     

        # Searching
        print('Performing Random search...')
        print('Pipeline:', [name for name, _ in pipeline.steps])
        start_time = t.time()
        rand_search.fit(self.X_train, self.y_train)                                   # find best hparameters using CV on training dataset
        end_time = t.time()
        print('Done in {:.3f}\n'.format((end_time - start_time)))
        
        # Results
        print('Cross-validation results:')
        results = pd.DataFrame(rand_search.cv_results_)
        columns = [col for col in results if col.startswith('mean') or col.startswith('std')]
        columns = [col for col in columns if 'time' not in col]
        results = results.sort_values(by='mean_test_'+refit_metric, ascending=False)
        results = results[columns].round(3).head()
        results.reset_index(drop=True, inplace=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):            # more options can be specified also
            print(results)

        # Best score
        print('\nBest cross-valid score: {:.2f}'.format(rand_search.best_score_*100))         # mean cross-validated score of the best_estimator (best hparam combination)

        # Best estimator
        best_estimator = rand_search.best_estimator_                          # estimator that was chosen by the search, i.e. estimator which gave highest score

        # Hparams of best estimator
        print('\nBest parameters set:')
        best_parameters = best_estimator.get_params()
        for param_name in sorted(hparams[0].keys()):
            print('\t{}: {}'.format(param_name, best_parameters[param_name]))

        # Create path and save model
        version = int(t.time())
        model_name = '{}_{}.joblib'.format(algorithm, version)
        model_path = os.path.join(PredictorManager.MODEL_DIR, model_name)
        joblib.dump(best_estimator, model_path)

        return best_estimator

    def evaluation_supervised(self, pipeline, algorithm, visual=True):
        """
        Evaluation
        """
        # Train eval
        y_preds = pipeline.predict(self.X_train)
        scores_train = f1_score(self.y_train, y_preds, average='macro')
        print('\n{} on train dataset: {:.2f}%'.format(algorithm, scores_train*100))
        
        # Test eval
        start_time = t.time()
        y_preds = pipeline.predict(self.X_test)
        end_time = t.time()
        scores_test = f1_score(self.y_test, y_preds, average='macro')
        print('{} on test dataset:  {:.2f}%\n'.format(algorithm, scores_test*100))
        
        # Report
        print(classification_report(self.y_test, y_preds), '\n')

        # Speed
        total_time = end_time - start_time
        single = total_time / len(self.y_test)
        ips = len(self.y_test) / total_time
        print('Total prediction time: \t{:.4f}s'.format(total_time))
        print('Single instance: \t{:.8f}s'.format(single))
        print('Instances per second: \t{:.2f}\n'.format(ips))
        
        # Visualization
        if visual:
            features = self.X_train.columns
            selected_features, estimator = PredictorManager.get_selected_features(features, pipeline)
            if selected_features is not features:
                print('Selected features:\n', selected_features)

            # Feature importance
            if algorithm in (Algorithms.RANDOM_FORSET):
                importances = estimator.feature_importances_
                PlotManager.feature_importance(selected_features, importances, title=algorithm)
            
            elif algorithm in (Algorithms.LOGISTIC_REGRESSION):
                classes = sorted(np.unique(self.y_test))
                for i, id_class in enumerate(classes):
                    importances = estimator.coef_[i]
                    title = 'Class {} - {}'.format(str(id_class), algorithm)
                    PlotManager.feature_importance(selected_features, importances, title=title)
        return

    @staticmethod
    def get_selected_features(features, pipeline):
        """
        Get selected features
            It works only in order Select K feature -> PCA transform
        """
        # No selection
        estimator = None
        selected_features = features

        # Get pipeline from calibartion wrapper
        if hasattr(pipeline, 'base_estimator'):
            pipeline = pipeline.base_estimator

        # Select features
        for step in pipeline.steps:
            name, transformer = step

            if name == 'kbest':
                scores = transformer.scores_
                k = transformer.get_params()['k']
                k_scores = pd.Series(scores).nlargest(k)
                selected_features = features[k_scores.index]

            elif name == 'pca' or name == 'lsa' or name == 'svd':
                n = transformer.get_params()['n_components']
                selected_features = np.arange(0, n)
                
            elif name == 'estimator':
                estimator = transformer

        return selected_features, estimator

    def optimize(self, algorithm, n_iters=3, n_folds=3, with_eval=True, visual=False):
        """
        Optimize model
        """
        # Get algorithm pipeline and hparam search space
        pipeline, hparams = Algorithms.get_alogirthm(algorithm=algorithm)

        # Random search
        best_estimator = self.random_search_supervised_cv(
            algorithm=algorithm, 
            pipeline=pipeline, 
            hparams=hparams,
            n_iters=n_iters,
            n_folds=n_folds)

        # Evaluation
        if with_eval:
            self.evaluation_supervised(best_estimator, algorithm, visual=visual)

        return best_estimator

    def load_model(self, path, algorithm, with_eval=True, visual=False):
        """
        Load model
        """
        estimator = joblib.load(path)

        if with_eval:
            self.evaluation_supervised(estimator, algorithm, visual=visual)

        return estimator


if __name__ == "__main__":

    predictor = PredictorManager(
        test_size=0.25,
        info=False
    )

    predictor.load_model(
        path='data/output/RandomForest_1633319603.joblib',
        algorithm=Algorithms.RANDOM_FORSET,
        with_eval=True,
        visual=True
    )

    predictor.optimize(
        algorithm=Algorithms.RANDOM_FORSET,
        n_iters=50,
        n_folds=5,
        with_eval=True,
        visual=True
    )

    print('done')