import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

class PolynomialRegressionPredictor:
    def __init__(self):
        self.model = None
        self.poly_features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def import_data(self, filepath):
        """Import datasets from a CSV file for scalar vectors."""
        data = pd.read_csv(filepath)
        return data

    def prepare_data(self, data, target_column):
        """Prepare features and target variable."""
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def create_and_fit_model(self, degree):
        """Create polynomial features and fit the linear regression model."""
        self.poly_features = PolynomialFeatures(degree=degree)
        X_poly_train = self.poly_features.fit_transform(self.X_train)
        
        self.model = LinearRegression()
        self.model.fit(X_poly_train, self.y_train)

    def test_model(self):
        """Test the model on the test dataset and return predictions."""
        X_poly_test = self.poly_features.transform(self.X_test)
        predictions = self.model.predict(X_poly_test)
        return predictions

    def evaluate_model(self, predictions):
        """Evaluate the model and return mean squared error and R-squared score."""
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        return mse, r2

    def hyperparameter_tuning(self):
        """Hyperparameter tuning using GridSearchCV for polynomial degree."""
        param_grid = {'degree': [1, 2, 3, 4, 5]}
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.poly_features.fit_transform(self.X_train), self.y_train)
        
        # Set the best degree
        best_degree = grid_search.best_params_['degree']
        self.create_and_fit_model(best_degree)
        return best_degree

    def visualize_data(self):
        """Visualize the fitted polynomial regression model."""
        plt.scatter(self.X_train, self.y_train, color='blue', label='Training Data')
        plt.scatter(self.X_test, self.y_test, color='red', label='Test Data')

        # Generate points for plotting the polynomial curve
        X_range = np.linspace(self.X_train.min(), self.X_train.max(), 100).reshape(-1, 1)
        X_range_poly = self.poly_features.transform(X_range)
        y_range = self.model.predict(X_range_poly)

        plt.plot(X_range, y_range, color='green', label='Polynomial Fit')
        plt.title('Polynomial Regression Fit')
        plt.xlabel('Features')
        plt.ylabel('Target')
        plt.legend()
        plt.show()

    def save_model(self, filename):
        """Save the model to a file."""
        dump((self.model, self.poly_features), filename)

# Example usage:
# predictor = PolynomialRegressionPredictor()
# data = predictor.import_data('path/to/your/scalar_vectors.csv')
# predictor.prepare_data(data, target_column='target_column_name')
# predictor.create_and_fit_model(degree=2)  # Initial fitting with degree 2
# predictions = predictor.test_model()
# mse, r2 = predictor.evaluate_model(predictions)
# best_degree = predictor.hyperparameter_tuning()
# predictor.visualize_data()
# predictor.save_model('polynomial_regression_model.joblib')
