from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

class mainpreprocessor:
    """
    Main preprocessor class that defines a series of transformers for preprocessing data.
    """

    class DropNA(BaseEstimator, TransformerMixin):
        """
        Transformer to drop rows with any missing values.
        """
        def fit(self, X, y=None):
            """
            Fit method (no fitting necessary for this transformer).
            """
            return self
        
        def transform(self, X):
            """
            Transform method to drop rows with any missing values.
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            return X.dropna()

    class RemoveDuplicatesWithSameCID(BaseEstimator, TransformerMixin):
        """
        Transformer to remove duplicate rows based on a specific column (CID).
        """
        def __init__(self, cid_column):
            """
            Initialize with the column name to check for duplicates.
            """
            self.cid_column = cid_column
        
        def fit(self, X, y=None):
            """
            Fit method (no fitting necessary for this transformer).
            """
            return self
        
        def transform(self, X):
            """
            Transform method to remove duplicate rows based on the specified column.
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            if self.cid_column not in X.columns:
                raise ValueError(f"Column '{self.cid_column}' not found in DataFrame")
            return X.drop_duplicates(subset=[self.cid_column])

    class DropDuplicates(BaseEstimator, TransformerMixin):
        """
        Transformer to drop duplicate rows.
        """
        def fit(self, X, y=None):
            """
            Fit method (no fitting necessary for this transformer).
            """
            return self
        
        def transform(self, X):
            """
            Transform method to drop duplicate rows.
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            return X.drop_duplicates()

    class KeepNecessaryColumns(BaseEstimator, TransformerMixin):
        """
        Transformer to keep only necessary columns.
        """
        def __init__(self):
            """
            Initialize with the list of necessary columns.
            """
            self.columns = [
                'mw', 'polararea', 'complexity', 'xlogp', 'heavycnt', 'hbonddonor', 'hbondacc', 'rotbonds', 'inchi',
                'exactmass', 'monoisotopicmass', 'charge', 'covalentunitcnt', 'isotopeatomcnt', 'totalatomstereocnt',
                'definedatomstereocnt', 'undefinedatomstereocnt', 'totalbondstereocnt', 'definedbondstereocnt',
                'undefinedbondstereocnt'
            ]
        
        def fit(self, X, y=None):
            """
            Fit method (no fitting necessary for this transformer).
            """
            return self
        
        def transform(self, X):
            """
            Transform method to keep only the necessary columns.
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            missing_columns = [col for col in self.columns if col not in X.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in DataFrame: {missing_columns}")
            return X[self.columns]

    class ScaleData(BaseEstimator, TransformerMixin):
        """
        Transformer to scale data (excluding 'inchi' column).
        """
        def __init__(self):
            """
            Initialize the scaler.
            """
            self.scaler = StandardScaler()
        
        def fit(self, X, y=None):
            """
            Fit method to fit the scaler on the data (excluding 'inchi' column).
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            if 'inchi' not in X.columns:
                raise ValueError("Column 'inchi' not found in DataFrame")
            self.scaler.fit(X.drop(columns=['inchi']))
            return self
        
        def transform(self, X):
            """
            Transform method to scale the data (excluding 'inchi' column).
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            if 'inchi' not in X.columns:
                raise ValueError("Column 'inchi' not found in DataFrame")
            scaled_data = pd.DataFrame(self.scaler.transform(X.drop(columns=['inchi'])), columns=X.columns.drop('inchi'))
            scaled_data['inchi'] = X['inchi'].values
            return scaled_data

    def __init__(self):
        """
        Initialize the main preprocessor with a defined preprocessing pipeline.
        """
        self.pipeline = Pipeline([
            ('dropna', mainpreprocessor.DropNA()),
            ('remove_duplicates_with_same_cid', mainpreprocessor.RemoveDuplicatesWithSameCID('cid')),
            ('drop_duplicates', mainpreprocessor.DropDuplicates()),
            ('keep_necessary_columns', mainpreprocessor.KeepNecessaryColumns()),
            ('scale_data', mainpreprocessor.ScaleData())
        ])

    def process(self, X):
        """
        Process the input DataFrame through the preprocessing pipeline.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        return self.pipeline.fit_transform(X)

class inhibitorprocessor(mainpreprocessor):
    """
    Inhibitor processor class that extends the main preprocessor and adds an 'inhibitor' column.
    """

    class AddInhibitorColumn(BaseEstimator, TransformerMixin):
        """
        Transformer to add an 'inhibitor' column with a specified value.
        """
        def __init__(self, value):
            """
            Initialize with the value to be added in the 'inhibitor' column.
            """
            self.value = value
        
        def fit(self, X, y=None):
            """
            Fit method (no fitting necessary for this transformer).
            """
            return self
        
        def transform(self, X):
            """
            Transform method to add the 'inhibitor' column with the specified value.
            """
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            X['inhibitor'] = self.value
            return X

    def __init__(self, inhibitor_value):
        """
        Initialize the inhibitor processor with the specified inhibitor value and append the AddInhibitorColumn transformer to the pipeline.
        """
        super().__init__()
        self.pipeline.steps.append(('add_inhibitor_column', inhibitorprocessor.AddInhibitorColumn(inhibitor_value)))

    def process(self, X=None):
        """
        Process the input DataFrame through the preprocessing pipeline and add the 'inhibitor' column.
        """
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            data = X
        else:
            raise ValueError("Input data must be provided.")
        
        return self.pipeline.fit_transform(data)