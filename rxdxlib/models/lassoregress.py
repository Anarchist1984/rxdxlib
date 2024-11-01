import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

class LassoInhibitorPredictor:
    def __init__(self, alpha=None):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importances_ = None

    def import_data(self, filepath):
        """Import dataset from a CSV file."""
        data = pd.read_csv(filepath)
        return data

    def prepare_data(self, data, target_column):
        """Prepare features and target variable."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def fit_model(self):
        """Fit the Lasso model with hyperparameter tuning."""
        self.model = LassoCV(cv=5, random_state=42).fit(self.X_train, self.y_train)

    def test_model(self):
        """Test the model on the test dataset."""
        predictions = self.model.predict(self.X_test)
        return predictions

    def evaluate_model(self, predictions):
        """Evaluate the model and return classification report and confusion matrix."""
        predicted_classes = self.threshold_predictions(predictions)
        report = classification_report(self.y_test, predicted_classes)
        cm = confusion_matrix(self.y_test, predicted_classes)
        return report, cm

    def threshold_predictions(self, predictions, threshold=0.5):
        """Threshold predictions to classify as inhibitor (1) or non-inhibitor (0)."""
        return np.where(predictions >= threshold, 1, 0)

    def get_feature_importances(self):
        """Examine coefficients to determine feature importance."""
        self.feature_importances_ = pd.Series(self.model.coef_, index=self.scaler.feature_names_in_)
        return self.feature_importances_

    def save_model(self, filename):
        """Save the model to a file."""
        dump(self.model, filename)

# Example usage:
# predictor = LassoInhibitorPredictor()
# data = predictor.import_data('path/to/your/dataset.csv')
# predictor.prepare_data(data, target_column='inhibitor_label')
# predictor.fit_model()
# predictions = predictor.test_model()
# report, cm = predictor.evaluate_model(predictions)
# importances = predictor.get_feature_importances()
# predictor.save_model('lasso_inhibitor_model.joblib')

