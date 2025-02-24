from Linear_Regression_S import LinearRegression_single
import numpy as np
from LinearRegression4 import X_test,y_test

mini_batch_model = LinearRegression_single()
mini_batch_model.load("regression4_mini_model_1.npz")

batch_model = LinearRegression_single()
batch_model.load("regression4_batch_model_1.npz")

stochastic_model = LinearRegression_single()
stochastic_model.load("regression4_stochastic_model_1.npz")

normal_equation_model = LinearRegression_single()
normal_equation_model.load("regression4_normal_equation_model_1.npz")

test_mse_loaded = mini_batch_model.score(X_test, y_test)
print(f"Test MSE (loaded model): {test_mse_loaded:.4f}")

predictions = mini_batch_model.predict( X_test
)
for pred, actual in zip(predictions[:5], y_test[:5]):
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}")