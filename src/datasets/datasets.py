__author__ = "alvrogd"


import typing

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.utils.data

import utils.constants as m_constants


class SeBASDataset(torch.utils.data.Dataset):

    def __init__(self, study_var: str, time_series: bool = True) -> None:

        super(SeBASDataset, self).__init__()

        self.study_var = study_var
        self.time_series = time_series


        # 0. Load data from disk
        data = pd.read_excel("data/dataset.xlsx")
            

        # 1. Pre-process data
        
        # Sort entries as the original script
        data = data.sort_values(by="yep")

        
        # 2. Extract target variable and its predictors
        
        # The "Year" col. is required to remove certain outliers in post-processing
        # The "ep" col. is required to group entries from the same plot when splitting the dataset
        data = data[["year", "ep", "explo", self.study_var] + m_constants.PREDICTORS]


        # 3. Post-process samples

        # Some samples may have missing values
        data = data.dropna()

        # Remove samples from plots with trees
        trees = ["AEG07", "AEG09", "AEG25", "AEG26", "AEG27", "AEG47", "AEG48", "HEG09", "HEG21", "HEG24", "HEG43"]
        data  = data[~data["ep"].isin(trees)]

        # Remove samples that were deemed outliers
        outliers2018 = ["SEG08", "SEG10", "SEG11", "SEG12", "SEG16", "SEG18", "SEG19", "SEG20", "SEG31", "SEG33",
                        "SEG35", "SEG36", "SEG38", "SEG39", "SEG40", "SEG41", "SEG44", "SEG46", "SEG45", "SEG49",
                        "SEG50"]
        data         = data[~((data["year"] == 2018) & (data["ep"].isin(outliers2018)))]

        # The year column is no longer needed
        data = data.drop(columns="year")

        # For the sake of mental sanity (when debugging), let's reset the indexes now that we have the final rows
        data = data.reset_index(drop=True)

        self.data            = data
        # The normalization will be performed per subset after splitting the dataset
        self.normalized_data = self.data.copy()


    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        # 1st col: plot ID
        # 2nd col: study variable
        # 3rd col onwards: predictors

        x = self.normalized_data.iloc[index, 3:]
        y = self.normalized_data.iloc[index, 2]

        x = x.to_numpy(np.float32)
        y = np.array([y], dtype=np.float32)
        
        if self.time_series:
            # Reshape the predictors to a time series of 16 steps and 10 bands
            x = x.reshape((10, 16), order="F")

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y
    
    
    def __len__(self) -> int:
        return len(self.normalized_data)


    def get_data(self) -> pd.DataFrame:
        return self.normalized_data


    def normalize_predictors(self, train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray) -> None:

        normalized_data = self.data.copy()
        scaler          = sklearn.preprocessing.StandardScaler()

        normalized_data.iloc[train_ids, normalized_data.columns.get_indexer(m_constants.PREDICTORS)] = \
            scaler.fit_transform(normalized_data.iloc[train_ids, normalized_data.columns.get_indexer(m_constants.PREDICTORS)])

        normalized_data.iloc[val_ids, normalized_data.columns.get_indexer(m_constants.PREDICTORS)] = \
            scaler.transform(normalized_data.iloc[val_ids, normalized_data.columns.get_indexer(m_constants.PREDICTORS)])

        normalized_data.iloc[test_ids, normalized_data.columns.get_indexer(m_constants.PREDICTORS)] = \
            scaler.transform(normalized_data.iloc[test_ids, normalized_data.columns.get_indexer(m_constants.PREDICTORS)])

        self.normalized_data = normalized_data


    def set_time_series(self, time_series: bool) -> None:
        self.time_series = time_series
        
      
def split_by_observatory(dataset: SeBASDataset, random_state: int) -> typing.List[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    exploratories = dataset.get_data()["explo"].unique()
    result        = []
    
    # Inside each fold, test on one observatory and train on the rest
    for i in range(len(exploratories)):
        
        test_exploratory    = exploratories[i]
        train_exploratories = exploratories[~(exploratories == test_exploratory)]
        
        train_ids = dataset.get_data().index[dataset.get_data()["explo"].isin(train_exploratories)].to_numpy()
        test_ids  = dataset.get_data().index[dataset.get_data()["explo"] == test_exploratory].to_numpy()
        
        # Split train IDs further into train and validation sets
        old_train_ids      = train_ids.copy()
        shuffle_splitter   = sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_ids, val_ids = next(shuffle_splitter.split(train_ids, groups=dataset.get_data()["ep"].iloc[train_ids]))
        train_ids, val_ids = old_train_ids[train_ids], old_train_ids[val_ids] 
        
        result.append((train_ids, val_ids, test_ids))
        
    return result
      
        
def split_by_plot(dataset: SeBASDataset, random_state: int) -> typing.List[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    kfold  = sklearn.model_selection.GroupKFold(n_splits=5)
    result = []
    
    # Group samples by plot ID inside folds to avoid spatial and temporal correlations between train and test data 
    for train_ids, test_ids in kfold.split(dataset, groups=dataset.get_data()["ep"]):
        
        # Split train IDs further into train and validation sets
        old_train_ids      = train_ids.copy()
        shuffle_splitter   = sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_ids, val_ids = next(shuffle_splitter.split(train_ids, groups=dataset.get_data()["ep"].iloc[train_ids]))
        train_ids, val_ids = old_train_ids[train_ids], old_train_ids[val_ids] 
        
        result.append((train_ids, val_ids, test_ids))
        
    return result        


# Needs to be in this module to avoid circular imports
SPLIT_PROCEDURES: typing.Dict[str, typing.Callable[[SeBASDataset, int], typing.List[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {
    "split_by_observatory": split_by_observatory,
    "split_by_plot":        split_by_plot,
}
