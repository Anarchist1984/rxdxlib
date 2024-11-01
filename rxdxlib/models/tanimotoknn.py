import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib

class TanimotoKNN:
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.knn = None
        self.best_k = None

    # Import inhibitors' fingerprints
    def import_inhibitors(self, smiles_inchi_list):
        self.inhibitor_fps = self.calculate_fingerprints(smiles_inchi_list)

    # Import possible non-inhibitors
    def import_non_inhibitors(self, smiles_inchi_list):
        self.non_inhibitor_fps = self.calculate_fingerprints(smiles_inchi_list)

    # Calculate fingerprints using RDKit
    def calculate_fingerprints(self, mol_list, input_type="SMILES"):
        fingerprints = []
        for mol_str in mol_list:
            if input_type == "SMILES":
                mol = Chem.MolFromSmiles(mol_str)
            elif input_type == "INCHI":
                mol = Chem.MolFromInchi(mol_str)  # Corrected the InChI function
            else:
                raise ValueError("Input type must be 'SMILES' or 'INCHI'")
                
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(fp)
        return fingerprints

    # Calculate Tanimoto distance between two fingerprints
    def tanimoto_distance(self, fp1, fp2):
        return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)

    # Define Similarity Threshold
    def set_similarity_threshold(self, threshold):
        self.similarity_threshold = threshold

    # Precompute Tanimoto distance matrix
    def precompute_tanimoto_matrix(self, X):
        n_samples = len(X)
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = self.tanimoto_distance(X[i], X[j])
        return distances

    # Cross-validation to determine optimal K value
    def select_k_value(self, X_train, y_train):
        param_grid = {'n_neighbors': list(range(1, 21))}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.best_k = grid_search.best_params_['n_neighbors']
        print(f"Best K value: {self.best_k}")

    # Train the model on actual inhibitor data
    def train(self, X_train, y_train):
        self.knn = KNeighborsClassifier(n_neighbors=self.best_k, metric='precomputed')
        tanimoto_matrix = self.precompute_tanimoto_matrix(X_train)
        self.knn.fit(tanimoto_matrix, y_train)

    # Run KNN on test data
    def predict(self, X_test):
        test_matrix = self.precompute_tanimoto_matrix(X_test)
        return self.knn.predict(test_matrix)

    # Label non-inhibitors with 0
    def label_non_inhibitors(self, predictions):
        return [0 if pred < self.similarity_threshold else 1 for pred in predictions]

    # Cross-validation score
    def cross_validate(self, X, y):
        tanimoto_matrix = self.precompute_tanimoto_matrix(X)
        scores = cross_val_score(self.knn, tanimoto_matrix, y, cv=5)
        print(f"Cross-validation scores: {scores}")
        return scores

    # Fine-tune similarity threshold
    def fine_tune_threshold(self, thresholds, X, y):
        best_threshold = None
        best_score = 0
        for threshold in thresholds:
            self.set_similarity_threshold(threshold)
            score = np.mean(self.cross_validate(X, y))
            if score > best_score:
                best_score = score
                best_threshold = threshold
        print(f"Best similarity threshold: {best_threshold}")

    # Visualization of similarity matrix
    def visualize_similarity(self, X):
        distances = self.precompute_tanimoto_matrix(X)
        plt.imshow(distances, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Tanimoto Similarity Heatmap")
        plt.show()

    # Save the model to disk
    def save_model(self, filename):
        joblib.dump(self.knn, filename)
        print(f"Model saved to {filename}")

