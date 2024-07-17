import pandas as pd
import os.path as osp
import torch
from sklearn.preprocessing import StandardScaler

from finbert.utils import InputExample

class TransactionsProcessor():

    def get_examples(self, data_dir, phase):
        """
        Get examples from the data directory.

        Parameters
        ----------
        data_dir: str
            Path for the data directory.
        phase: str
            Name of the .csv file to be loaded.
        """
        data = pd.read_csv(osp.join(data_dir, phase + ".csv"))
        self.labels = data["category"].unique().tolist()
        data = data.reset_index()[["index", "description", "category"]]
        return self._create_examples(data.values)
    
    def get_labels(self):
        return self.labels
    
    def _create_examples(self, data_values):
        return [InputExample(i, description, category)
                for i, description, category in data_values]
    
    def get_numeric(self, data_dir, phase, text_colname="description", label_colname="category", ret_tensor=True):
        ## ! IMPORTANT ASSUMPTION THAT DATA IS ALREADY SCALED
        data = pd.read_csv(osp.join(data_dir, phase + ".csv"))
        numeric_feats = data[[col for col in data.columns if col not in [text_colname, label_colname]]]
        if ret_tensor:
            numeric_feats = torch.from_numpy(numeric_feats.values.astype(float))
        return numeric_feats