
import time as t
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import scipy.stats as stats

from mngrplot import PlotManager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


class DataManager:
    """
    Data manager
    """

    RAW_DATA_PATH = 'data/input/ffbdata.csv'
    DATA_PATH = 'data/input/ffdata_final.csv'
    RANDOM_STATE = 21
    CLASS_LABEL = 'status_num'

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        return

    @staticmethod
    def describe_data(df, info=True, duplicates=True):
        """
        Describe given dataframe
            Source: https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/getting_started.html
                    https://github.com/ResidentMario/missingno
        """
        if info:
            # Dataset
            print(df)

            # NaNs
            print('NaN:\n', df.isna().sum())
            print('NaN rows:\n', df[df.isna().any(axis=1)])
            print(df.describe(include='all'))

            # Visualization
            msno.matrix(df)
            plt.show()
            plt.close()

            # Duplicates
            if duplicates:
                for col in df.columns:
                    uni_val = df[col].value_counts()
                    uni_num = len(uni_val)
                    print('{} : {}\n{}\n'.format(col, uni_num, uni_val))
        
                dupl = df.duplicated().value_counts()
                #df = df.drop_duplicates(keep='first')
                print('Duplicates:\n', dupl)
        return

    def read_dataset(self, path):
        """
        Read dataset
        """
        dataset = pd.read_csv(path, header=0)
        return dataset

    def clean_dataset(self, remove=True, info=True):
        """
        Clean given raw dataset
        """
        # Read raw dataset
        dataset = self.read_dataset(path=DataManager.RAW_DATA_PATH)

        # Info
        DataManager.describe_data(dataset)

        # Merge state and country
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='_')
        dataset['state'] = imp.fit_transform(dataset[['state']])
        dataset['user_state'] = imp.fit_transform(dataset[['user_state']])
        dataset['country_state'] = dataset['country'] + '_' + dataset['state']
        dataset['user_country_state'] = dataset['user_country'] + '_' + dataset['user_state']
        if info:
            DataManager.describe_data(dataset[['country_state', 'user_country_state']])
            DataManager.describe_data(dataset)

        # Drop rows which contains empty value for given list of features
        if remove:
            columns_drop_rows = [
                'lake_fishing',
                'river_fishing',
                'inshore_fishing',
                'offshore_fishing',
                'big_game_fishing',
                'bottom_fishing',
                'trolling',
                'light_tackle',
                'heavy_tackle',
                'fly_fishing',
                'jigging',
                'lunch_included',
                'snacks_included',
                'drinks_included'
            ]
            for col in columns_drop_rows:
                rows = dataset[dataset[col].isna()]
                dataset = dataset.drop(rows.index)
                if info:
                    print(rows)
            DataManager.describe_data(dataset)

        # Impute constant into missing values (train + test)
        else:
            columns_missing_const = [
                'lake_fishing',
                'river_fishing',
                'inshore_fishing',
                'offshore_fishing',
                'big_game_fishing',
                'bottom_fishing',
                'trolling',
                'light_tackle',
                'heavy_tackle',
                'fly_fishing',
                'jigging',
                'lunch_included',
                'snacks_included',
                'drinks_included'
            ]
            for col in columns_missing_const:
                imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
                curr_col = dataset[[col]]
                dataset[col] = imp.fit_transform(curr_col)
            if info:
                DataManager.describe_data(dataset)

            # Remove rows that don't contain none of the charter opportunities
            non_charter = dataset[columns_missing_const]
            non_charter['valid_charter'] = non_charter.sum(axis=1).values
            rows = non_charter[non_charter['valid_charter'] == 0]
            if len(rows) > 0:
                dataset = dataset.drop(rows.index)
                if info:
                    print(rows)
                    print(rows.index)
                    DataManager.describe_data(dataset)

        # Convert categorical to numeric
        columns_str = [
            'status',
            'user_city',
            'user_country_state',
            'location',
            'country_state',
            'charter_title',
            'package_title',
            'departure_time'
        ]
        for col in columns_str:
            ord_encoder = OrdinalEncoder()
            col_name = '{}_num'.format(col)
            curr_col = dataset[[col]]
            dataset[col_name] = ord_encoder.fit_transform(curr_col)

        # Convert datetime
        columns_date = ['trip_date', 'date_created']
        for col in columns_date:
            dataset[col] = pd.to_datetime(dataset[col])
        dataset['trip_time_diff'] = pd.to_numeric(dataset['trip_date'] - dataset['date_created']) / 1E12

        # Save clean version
        DataManager.describe_data(dataset)
        dataset.to_csv(DataManager.DATA_PATH, index=False, date_format='%Y-%m-%d')
        return dataset

    def set_types(self, dataset):
        """
        Set types
        """
        # Datatime type
        columns_date = ['trip_date', 'date_created']
        for col in columns_date:
            dataset[col] = pd.to_datetime(dataset[col])

        # Int type
        dataset[DataManager.CLASS_LABEL] = dataset[DataManager.CLASS_LABEL].astype(int)

        return dataset

    def filter_dataset(self, dataset):
        """
        Return new dataset (copy) with filter columns
        """
        columns = [
            'charter_id',
            'package_id',
            'user_id',
            'person_count',
            'children_count',
            'instantly_booked',
            'trip_time_diff',
            'captain_id',
            'public',
            'anglers_choice_award',
            'lake_fishing',
            'river_fishing',
            'inshore_fishing',
            'offshore_fishing',
            'big_game_fishing',
            'bottom_fishing',
            'trolling',
            'light_tackle',
            'heavy_tackle',
            'fly_fishing',
            'jigging',
            'lunch_included',
            'snacks_included',
            'drinks_included',
            'price',
            'shared',
            'seasonal',
            'duration_hours',
            'user_city_num',
            'user_country_state_num',
            'location_num',
            'country_state_num',
            'charter_title_num',
            'package_title_num',
            'departure_time_num',
            DataManager.CLASS_LABEL
        ]
        dataset = dataset[columns].copy()
        rows = dataset[dataset[DataManager.CLASS_LABEL] == 2].tail(7000)
        #rows = dataset[dataset[DataManager.CLASS_LABEL] == 1]
        dataset = dataset.drop(rows.index)
        #DataManager.describe_data(dataset)
        
        return dataset

    def split_dataset(self, dataset, test_size):
        """
        Split dataset for training, validation and testing
        """
        # Split
        X_data = dataset.iloc[:, :-1]
        y_data = dataset.iloc[:, -1:]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_data, y_data, test_size=test_size, stratify=y_data, random_state=DataManager.RANDOM_STATE)
        
        # Info
        print('\nTraining set: {:.2f}% ({})'.format(len(self.y_train) / len(y_data) * 100, len(self.y_train)))
        print('\nDistribution:', self.y_train.value_counts().sort_index(), sep='\n')
        print('\nTesting set:  {:.2f}% ({})'.format(len(self.y_test) / len(y_data) * 100, len(self.y_test)))
        print('\nDistribution:', self.y_test.value_counts().sort_index(), sep='\n')
        return

    def get_dataset(self, test_size=0.1, info=False):
        """
        Get clean dataset
        """
        # Clean dataset with details
        self.dataset = self.read_dataset(path=DataManager.DATA_PATH)
        self.dataset = self.set_types(self.dataset)
        if info:
            DataManager.describe_data(self.dataset)
            #PlotManager.visualize_data(self.dataset)

        # Dataset for predictions
        dataset_ml = self.filter_dataset(self.dataset)
        if info:
            DataManager.describe_data(dataset_ml)
            PlotManager.visualize_data(dataset_ml)

        self.split_dataset(dataset_ml, test_size)
        return self.dataset


if __name__ == "__main__":

    datamngr = DataManager()
    
    #datamngr.clean_dataset(remove=False)

    datamngr.get_dataset(
        test_size=0.1,
        info=True)