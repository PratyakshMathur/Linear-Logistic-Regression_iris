import numpy as np
from numpy.linalg import inv
import os
import time

class LinearRegression_single:
    def __init__(self):
        self.weight = None  # Scalar weights (list for multiple features)
        self.bias = 0  # Scalar bias
        self.best_weight = None
        self.best_bias = 0
        self.best_val_loss = float('inf')

    def fit(self, feature, target, method='gradient_descent', gradient_method='m', 
            batch_size=32, learning_rate=0.01, regularization=0, max_epochs=100, patience=3):
        start_time = time.time()
        feature = np.array(feature)
        target = np.array(target).flatten()
        n_samples, n_features = feature.shape
        
        n_val = int(0.1 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        feature_train, target_train = feature[train_indices], target[train_indices]
        feature_val, target_val = feature[val_indices], target[val_indices]

        if method == 'gradient_descent':
            self.weight = np.random.randn(n_features) * 0.01
            self.bias = np.random.randn() * 0.01
            
            if gradient_method == 's':
                batch_size = 1
            elif gradient_method == 'b':
                batch_size = n_samples
            elif gradient_method != 'm':
                raise ValueError("gradient_method must be 's', 'b', or 'm'")
            
            self.best_weight, self.best_bias = self.gradient_descent(
                feature_train, target_train, feature_val, target_val, self.weight, self.bias,feature_train.shape[0],
                gradient_method, batch_size, learning_rate, max_epochs, regularization, patience
            )
            
        
        elif method == 'normal_equation':
            self.best_weight, self.best_bias = self.normal_equation(feature_train, target_train, regularization)
        else:
            raise ValueError("method must be 'gradient_descent' or 'normal_equation'")
        
        end_time = time.time()
        training_time = end_time - start_time
        return self.best_weight, self.best_bias, training_time
    
    def gradient_descent(self, X, y, feature_val, target_val, weight, bias, n_samples, approach, batch_size,
                        learning_rate=0.01, max_epochs=1000, reg_lambda=0, patience=10, tol=1e-6):
        patience_counter = 0
        
        for epoch in range(max_epochs):
            if approach == 's':  # Stochastic Gradient Descent
                for i in range(n_samples):
                    yi = y[i]
                    y_pred = bias
                    for j in range(len(weight)):
                        y_pred += weight[j] * X[i][j]
                    error = y_pred - yi
                    for j in range(len(weight)):
                        dW = X[i][j] * error + reg_lambda * weight[j]
                        weight[j] -= learning_rate * dW
                    bias -= learning_rate * error
            else:
                indices = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[indices], y[indices]
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    batch_size_actual = len(y_batch)
                    
                    y_pred_batch = [bias] * batch_size_actual
                    for j in range(len(weight)):
                        for k in range(batch_size_actual):
                            y_pred_batch[k] += weight[j] * X_batch[k][j]
                    
                    errors = [y_pred_batch[k] - y_batch[k] for k in range(batch_size_actual)]
                    
                    for j in range(len(weight)):
                        dW = sum(X_batch[k][j] * errors[k] for k in range(batch_size_actual)) / batch_size_actual + reg_lambda * weight[j]
                        weight[j] -= learning_rate * dW
                    
                    db = sum(errors) / batch_size_actual
                    bias -= learning_rate * db
            
            val_loss = 0
            for k in range(len(feature_val)):
                val_pred = bias
                for j in range(len(weight)):
                    val_pred += weight[j] * feature_val[k][j]
                val_loss += (val_pred - target_val[k]) ** 2
            val_loss /= len(feature_val)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weight = weight[:]
                self.best_bias = bias
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    return self.best_weight, self.best_bias
        
        return self.best_weight, self.best_bias
    
    def normal_equation(self, X, y, reg_lambda=0):
        X = np.c_[np.ones(X.shape[0]), X]
        I = np.eye(X.shape[1])
        W = inv(X.T @ X + reg_lambda * I) @ X.T @ y
        return W[1:], W[0]
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            y_pred = self.best_bias
            for j in range(len(self.best_weight)):
                y_pred += self.best_weight[j] * X[i][j]
            predictions.append(y_pred)
        return np.array(predictions)

    
    def score(self, X, y):
        """Calculate Mean Squared Error between predictions and target values."""
        X = np.array(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        m_outputs = y.shape[1] if y.ndim > 1 else 1
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.predict(X)
        mse = np.sum((y - y_pred) ** 2) / (n_samples * m_outputs)
        
        return mse
    
    def save(self, filepath):

        # Ensure the filepath has .npz extension
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare model parameters
        model_params = {
            'weights': self.best_weight,
            'bias': self.best_bias,
            
        }
        
        try:
            # Save the parameters
            np.savez(filepath, **model_params)
            print(f"Model parameters saved successfully to {filepath}")
        except Exception as e:
            raise Exception(f"Error saving model parameters: {str(e)}")
    
    def load(self, filepath):
 
        # Ensure the filepath has .npz extension
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        try:
            # Load the parameters
            loaded = np.load(filepath)
            
            # Set model parameters
            self.best_weight = loaded['weights']
            self.best_bias - loaded['bias']
            
            print(f"Model parameters loaded successfully from {filepath}")
        except Exception as e:
            raise Exception(f"Error loading model parameters: {str(e)}")