from Linear_Regression import LinearRegression
import numpy as np
from train_LinearRegression3 import X_test,y_test

mini_batch_model = LinearRegression()
mini_batch_model.load("models/regression2_mini_model_1.npz")

m_loaded_acc = mini_batch_model.score(X_test, y_test)
print(f"Mini-Batch MSE (loaded model): {m_loaded_acc:.4f}")

batch_model = LinearRegression()
batch_model.load("models/regression2_batch_model_1.npz")

b_loaded_acc = batch_model.score(X_test, y_test)
print(f"Batch MSE (loaded model): {b_loaded_acc:.4f}")

stochastic_model = LinearRegression()
stochastic_model.load("models/regression2_stochastic_model_1.npz")

s_loaded_acc = stochastic_model.score(X_test, y_test)
print(f"Stocastic MSE (loaded model): {s_loaded_acc:.4f}")

normal_equation_model = LinearRegression()
normal_equation_model.load("models/regression2_normal_equation_model_1.npz")

n_loaded_acc = normal_equation_model.score(X_test, y_test)
print(f"Natural-Equation MSE (loaded model): {n_loaded_acc:.4f}")


best = [mini_batch_model, batch_model, stochastic_model, normal_equation_model][np.argmin([m_loaded_acc, b_loaded_acc, s_loaded_acc, n_loaded_acc])]
best.save("models/best_regression_model_2.npz")


best_model = LinearRegression()
best_model.load("models/best_regression_model_2.npz")

test_mse_loaded = best_model.score(X_test, y_test)
print(f"Best MSE (loaded model): {test_mse_loaded:.4f}")

predictions = best_model.predict( X_test
)
for pred, actual in zip(predictions[:5], y_test[:5]):
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}")