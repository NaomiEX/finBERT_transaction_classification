import pandas as pd
import os.path as osp
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