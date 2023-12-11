import tarfile
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm
import joblib
# Download the CIFAR-10 taz.gz file https://www.cs.toronto.edu/~kriz/cifar.html
# K-Nearest Neighbors Classifier
class KNNClassifier:
    def __init__(self, k=1):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point in tqdm(X_test, desc='Classifying', ncols=100):
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions

    def predict_parallel(self, X_test):
        predictions = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.predict_point)(test_point) for test_point in tqdm(X_test, desc='Classifying', ncols=100)
        )
        return predictions

    def predict_point(self, test_point):
        distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        return most_common_label


# Nearest Centroid Classifier
class NearestCentroid:
    def __init__(self):
        self.centroids = None

    def fit(self, X, y):
        unique_classes = np.unique(y)
        self.centroids = []

        for class_label in unique_classes:
            class_indices = np.where(y == class_label)
            class_data = X[class_indices]
            class_centroid = np.mean(class_data, axis=0)
            self.centroids.append(class_centroid)

        self.centroids = np.array(self.centroids)

    def predict(self, X):
        predictions = []
        for sample in X:
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            predicted_class = np.argmin(distances)
            predictions.append(predicted_class)
        return np.array(predictions)


# Function to calculate classification accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


# Function to unpickle data files
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


# Load CIFAR-10 data



def load_train_test_batches(train_files, test_file):
    train_data_batches = []
    train_labels_batches = []

    for train_file in train_files:
        batch = unpickle(train_file)
        train_data_batch = np.array(batch[b'data'])
        train_labels_batch = np.array(batch[b'labels'])

        train_data_batches.append(train_data_batch)
        train_labels_batches.append(train_labels_batch)

    test_batch = unpickle(test_file)
    test_data = np.array(test_batch[b'data'])
    test_labels = np.array(test_batch[b'labels'])

    train_data = np.concatenate(train_data_batches, axis=0)
    train_labels = np.concatenate(train_labels_batches, axis=0)

    return train_data, train_labels, test_data, test_labels


# List of data files for training batches (data_batch_1 to data_batch_5) and test batch (test_batch)
train_files = ['cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1, 6)]
test_file = 'cifar-10-batches-py/test_batch'

# Load data from training and test batches
X_train, y_train, X_test, y_test = load_train_test_batches(train_files, test_file)

# Instantiate the KNNClassifier with k=1
knn_classifier = KNNClassifier(k=1)
knn_classifier.fit(X_train, y_train)
predictions1 = knn_classifier.predict_parallel(X_test)
nn1_accuracy = calculate_accuracy(y_test, predictions1)

# Instantiate the KNNClassifier with k=3
knn_classifier = KNNClassifier(k=3)
knn_classifier.fit(X_train, y_train)
predictions3 = knn_classifier.predict_parallel(X_test)
nn3_accuracy = calculate_accuracy(y_test, predictions3)

# Nearest-Center Categorizer
nc_classifier = NearestCentroid()
nc_classifier.fit(X_train, y_train)
nc_predictions = nc_classifier.predict(X_test)
nc_accuracy = calculate_accuracy(y_test, nc_predictions)

# Print the accuracies
print("Accuracy - Nearest Neighbor (1 neighbor): {:.2f}%".format(nn1_accuracy * 100))
print("Accuracy - Nearest Neighbor (3 neighbors): {:.2f}%".format(nn3_accuracy * 100))
print("Accuracy - Nearest-Center Classifier: {:.2f}%".format(nc_accuracy * 100))
