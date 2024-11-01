import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

class RidgeInhibitorPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def import_data(self, filepath_fingerprints, filepath_scalars):
        """Import datasets from CSV files for fingerprints and scaled scalar data."""
        fingerprints = pd.read_csv(filepath_fingerprints)
        scalars = pd.read_csv(filepath_scalars)
        return fingerprints, scalars

    def prepare_data(self, fingerprints, scalars, target_column):
        """Prepare features and target variable by merging fingerprints and scaled scalar data."""
        X = pd.concat([fingerprints, scalars], axis=1)
        y = scalars[target_column]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def fit_model(self):
        """Fit the Ridge model."""
        self.model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5).fit(self.X_train, self.y_train)

    def test_model(self):
        """Test the model on the test dataset and return predictions."""
        predictions = self.model.predict(self.X_test)
        return predictions

    def evaluate_model(self, predictions):
        """Evaluate the model and return mean squared error and R-squared score."""
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        return mse, r2

    def refit_with_optimal_alpha(self):
        """Refit the model using the optimal alpha."""
        optimal_alpha = self.model.alpha_
        self.model = RidgeCV(alphas=[optimal_alpha]).fit(self.X_train, self.y_train)

    def save_model(self, filename):
        """Save the model to a file."""
        dump(self.model, filename)

# Example usage:
# predictor = RidgeInhibitorPredictor()
# fingerprints, scalars = predictor.import_data('path/to/fingerprints.csv', 'path/to/scalars.csv')
# predictor.prepare_data(fingerprints, scalars, target_column='inhibitor_label')
# predictor.fit_model()
# predictions = predictor.test_model()
# mse, r2 = predictor.evaluate_model(predictions)
# predictor.refit_with_optimal_alpha()
# predictor.save_model('ridge_inhibitor_model.joblib')
