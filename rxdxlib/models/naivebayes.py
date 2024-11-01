import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load

class NaiveBayesPredictor:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def import_data(self, filepath_combined, filepath_fingerprints, filepath_scalars):
        """Import datasets from CSV files."""
        combined_data = pd.read_csv(filepath_combined)
        fingerprints = pd.read_csv(filepath_fingerprints)
        scalars = pd.read_csv(filepath_scalars)
        return combined_data, fingerprints, scalars

    def prepare_data(self, combined_data, fingerprints, scalars, target_column):
        """Prepare features and target variable."""
        X = pd.concat([combined_data, fingerprints, scalars], axis=1)
        y = combined_data[target_column]
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def select_model(self, feature_type):
        """Select the appropriate Naive Bayes model based on the feature type."""
        if feature_type == 'continuous':
            self.model = GaussianNB()
        elif feature_type == 'binary':
            self.model = BernoulliNB()
        elif feature_type == 'discrete':
            self.model = MultinomialNB()
        else:
            raise ValueError("Unsupported feature type. Choose 'continuous', 'binary', or 'discrete'.")

    def fit_model(self):
        """Fit the selected Naive Bayes model."""
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        """Test the model on the test dataset and return predictions."""
        predictions = self.model.predict(self.X_test)
        return predictions

    def evaluate_model(self, predictions):
        """Evaluate the model and return classification report and accuracy score."""
        report = classification_report(self.y_test, predictions)
        accuracy = accuracy_score(self.y_test, predictions)
        return report, accuracy

    def experiment_with_smoothing(self, alpha_values):
        """Experiment with different smoothing parameters (alpha) for Bernoulli and Multinomial Naive Bayes."""
        results = {}
        for alpha in alpha_values:
            if isinstance(self.model, (BernoulliNB, MultinomialNB)):
                self.model.set_params(alpha=alpha)
                self.fit_model()
                predictions = self.test_model()
                report, accuracy = self.evaluate_model(predictions)
                results[alpha] = accuracy  # Store accuracy for each alpha
        return results

    def save_model(self, filename):
        """Save the model to a file."""
        dump(self.model, filename)

# Example usage:
# predictor = NaiveBayesPredictor()
# combined_data, fingerprints, scalars = predictor.import_data('path/to/combined.csv', 'path/to/fingerprints.csv', 'path/to/scalars.csv')
# predictor.prepare_data(combined_data, fingerprints, scalars, target_column='target_column_name')
# predictor.select_model(feature_type='continuous')  # Choose feature type: 'continuous', 'binary', or 'discrete'
# predictor.fit_model()
# predictions = predictor.test_model()
# report, accuracy = predictor.evaluate_model(predictions)
# smoothing_results = predictor.experiment_with_smoothing(alpha_values=[0.1, 0.5, 1.0, 1.5, 2.0])
# predictor.save_model('naive_bayes_model.joblib')
