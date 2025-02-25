import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Linear_Regression import LinearRegression

# Load the Iris dataset
iris = load_iris()
# print(iris)
X = iris.data[:, [0, 1, 3]]
y = iris.data[:, [2]]  # Predict petal length using other features


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=iris.target  # Ensure even split of classes
)
def main():
    time_taken = {}
    # Create and train the model
    mini_batch_model = LinearRegression()

    weights , time_m = mini_batch_model.fit(
        feature=X_train,
        target=y_train,
        method='gradient_descent',
        gradient_method='m',
        batch_size=32,
        learning_rate=0.01,
        regularization=0.001,
        max_epochs=100,
        patience=3
    )
    time_taken['m'] = time_m
    mini_batch_model.plot_val_loss("Plot-3-mini-batch")

    batch_model = LinearRegression()

    weights ,time_b = batch_model.fit(
        feature=X_train,
        target=y_train,
        method='gradient_descent',
        gradient_method='b',
        learning_rate=0.01,
        regularization=0.01,
        max_epochs=1000,
        patience=100
    )
    time_taken['b'] = time_b
    batch_model.plot_val_loss("Plot-3-batch-batch")

    stochastic_model = LinearRegression()

    weights ,time_s = stochastic_model.fit(
        feature=X_train,
        target=y_train,
        method='gradient_descent',
        gradient_method='s',
        learning_rate=0.02,
        regularization=0.01,
        max_epochs=1000,
        patience=100
    )
    time_taken['s'] = time_s
    stochastic_model.plot_val_loss("Plot-3-stocastic-batch")
    normal_equation_model = LinearRegression()

    weights , time_n = normal_equation_model.fit(
        feature=X_train,
        target=y_train,
        method='normal_equation',
        learning_rate=0.01,
        regularization=0.01,
    )
    time_taken['n'] = time_n
    #Save the model
    mini_batch_model.save('models/regression3_mini_model_1.npz')
    batch_model.save('models/regression3_batch_model_1.npz')
    stochastic_model.save('models/regression3_stochastic_model_1.npz')
    normal_equation_model.save('models/regression3_normal_equation_model_1.npz')

    # Print final training score
    model_m_score = mini_batch_model.score(X_train, y_train)
    print(f"Final training MSE Mini-Batch: {model_m_score:.4f}")

    model_b_score = batch_model.score(X_train, y_train)
    print(f"Final training MSE Batch: {model_b_score:.4f}")

    model_s_score = stochastic_model.score(X_train, y_train)
    print(f"Final training MSE Stochastic: {model_s_score:.4f}")

    model_n_score = normal_equation_model.score(X_train, y_train)
    print(f"Final training MSE Normal-Equation: {model_n_score:.4f}")
    print(time_taken)

if __name__ == "__main__":
    main()