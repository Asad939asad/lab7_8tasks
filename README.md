As the accuracy on the model trained on logistic regression was 85 %
Data Generation:

make_classification generates a random n-class classification problem.

n_samples=1000 specifies that we want 1000 samples.
n_features=20 indicates that there are 20 features in the dataset.
random_state=42 ensures reproducibility.


Data Splitting:
The dataset is split into training and testing sets using train_test_split.
test_size=0.2 means 20% of the data is used for testing, and 80% for training.
random_state=42 ensures consistent splitting across runs.


Model Initialization:
A Logistic Regression classifier (model) is initialized.
max_iter=1000 sets the maximum number of iterations for the solver.
random_state=42 ensures consistent results.


Model Training:
The Logistic Regression model (model) is trained on the training data (X_train, y_train) using fit.


Prediction:
Predictions (y_pred) are made on the testing data (X_test) using predict.


Accuracy Calculation:
The accuracy of the model is calculated using accuracy_score by comparing the predicted labels (y_pred) with the actual labels (y_test).
The accuracy percentage is then printed to the console.



The aaccuracy obtained on model trained with random forest classifier is 90%

Data Generation:
make_classification generates a random n-class classification problem.
n_samples=1000 specifies that we want 1000 samples.
n_features=20 indicates that there are 20 features in the dataset.
random_state=42 ensures reproducibility.


Data Splitting:
The dataset is split into training and testing sets using train_test_split.
test_size=0.2 means 20% of the data is used for testing, and 80% for training.
random_state=42 ensures consistent splitting across runs.


Feature Scaling:
StandardScaler standardizes features by removing the mean and scaling to unit variance.
It's applied to both training and testing data separately to avoid data leakage.


Model Initialization:
A Random Forest classifier is initialized with n_estimators=100, meaning it will use 100 decision trees.
random_state=42 ensures consistent results.


Model Training:
The Random Forest classifier (clf) is trained on the scaled training data using fit.


Prediction:
Predictions (y_pred) are made on the scaled testing data using predict.


Accuracy Calculation:
The accuracy of the model is calculated using accuracy_score by comparing the predicted labels (y_pred) with the actual labels (y_test).
The accuracy percentage is then printed to the console.
