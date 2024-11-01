import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

class SVMInhibitorPredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def import_data(self, filepath_vector, filepath_labels):
        """Import datasets from CSV files for features (singular vector) and labels."""
        vectors = pd.read_csv(filepath_vector)
        labels = pd.read_csv(filepath_labels)
        return vectors, labels

    def prepare_data(self, vectors, labels, target_column):
        """Prepare features and target variable."""
        X = vectors
        y = labels[target_column]
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def create_and_train_model(self):
        """Create and train the SVM model."""
        self.model = SVC(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        """Test the model on the test dataset and return predictions."""
        predictions = self.model.predict(self.X_test)
        return predictions

    def evaluate_model(self, predictions):
        """Evaluate the model and return classification report and confusion matrix."""
        report = classification_report(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)
        return report, cm

    def hyperparameter_tuning(self):
        """Hyperparameter tuning using GridSearchCV."""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly']
        }
        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Set the best parameters
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def save_model(self, filename):
        """Save the model to a file."""
        dump(self.model, filename)

# Example usage:
# predictor = SVMInhibitorPredictor()
# vectors, labels = predictor.import_data('path/to/singular_vectors.csv', 'path/to/labels.csv')
# predictor.prepare_data(vectors, labels, target_column='inhibitor_label')
# predictor.create_and_train_model()
# predictions = predictor.test_model()
# report, cm = predictor.evaluate_model(predictions)
# best_params = predictor.hyperparameter_tuning()
# predictor.save_model('svm_inhibitor_model.joblib')
