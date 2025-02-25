import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Logistic_Regression import LogisticRegression



# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names=iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

def main():

    # 1. Stochastic Gradient Descent
    print("Training with Stochastic Gradient Descent...")
    sgd_model = LogisticRegression()
    sgd_model.fit(
        feature=X_train,
        target=y_train,
        method='gradient_descent',
        gradient_method='s',
        learning_rate=0.01,
        regularization=0.01,
        max_epochs=100
    )
    sgd_acc = sgd_model.score(X_test, y_test)
    print(f"SGD Accuracy: {sgd_acc:.4f}\n")
    sgd_model.save("models/classification_stochastic_model_1.npz")

    # 2. Batch Gradient Descent
    print("Training with Batch Gradient Descent...")
    bgd_model = LogisticRegression()
    bgd_model.fit(
        feature=X_train,
        target=y_train,
        method='gradient_descent',
        gradient_method='b',
        learning_rate=0.01,
        regularization=0.01,
        max_epochs=100
    )
    bgd_acc = bgd_model.score(X_test, y_test)
    print(f"BGD Accuracy: {bgd_acc:.4f}\n")
    bgd_model.save("models/classification_batch_model_1.npz")

    # 3. Mini-Batch Gradient Descent
    print("Training with Mini-Batch Gradient Descent...")
    mbgd_model = LogisticRegression()
    mbgd_model.fit(
        feature=X_train,
        target=y_train,
        method='gradient_descent',
        gradient_method='m',
        batch_size=32,
        learning_rate=0.01,
        regularization=0.01,
        max_epochs=100
    )
    mbgd_acc = mbgd_model.score(X_test, y_test)
    print(f"Mini-Batch GD Accuracy: {mbgd_acc:.4f}\n")
    mbgd_model.save("models/classification_mini_model_1.npz")

    # 4. Normal Equation
    print("Training with Normal Equation (IRLS)...")
    ne_model = LogisticRegression()
    ne_model.fit(
        feature=X_train,
        target=y_train,
        method='normal_equation',
        regularization=0.01,
        max_epochs=100
    )
    ne_acc = ne_model.score(X_test, y_test)
    print(f"Normal Equation Accuracy: {ne_acc:.4f}\n")
    ne_model.save("models/classification_normal_model_1.npz")


if __name__ == '__main__':
    main()