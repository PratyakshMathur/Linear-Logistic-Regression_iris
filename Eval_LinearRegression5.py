from Linear_Regression_M import LinearRegression_multiple
import numpy as np
from LinearRegression5 import X_test,y_test

mini_batch_model = LinearRegression_multiple()
mini_batch_model.load("regression_multiple_mini_model.npz")

batch_model = LinearRegression_multiple()
batch_model.load("regression_multiple_batch_model.npz")

stochastic_model = LinearRegression_multiple()
stochastic_model.load("regression_multiple_stochastic_model.npz")

normal_equation_model = LinearRegression_multiple()
normal_equation_model.load("regression_multiple_normal_equation_model.npz")

test_mse_loaded = batch_model.score(X_test, y_test)
print(f"Test MSE (loaded model): {test_mse_loaded:.4f}")

predictions = batch_model.predict( X_test
)
print(predictions.shape) 
for pred, actual in zip(predictions[:5], y_test[:5]):
    print(f"Predicted: {pred[0]:.2f}, {pred[1]:.2f}, Actual: {actual[0]:.2f}, {actual[1]:.2f}")