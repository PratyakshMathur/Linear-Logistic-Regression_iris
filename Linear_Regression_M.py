import numpy as np
from numpy.linalg import inv
import os
import time


class LinearRegression_multiple:
    def __init__(self):
        # Initialize model parameters
        self.weights = None
        self.best_weights = None
        self.best_val_loss = float('inf')
        
    
    def fit(self, feature, target, method='gradient_descent', gradient_method='m', 
            batch_size=32, learning_rate=0.01, regularization=0, max_epochs=100, patience=3):
        """Train the Linear Regression model for multiple outputs and record training time."""
        
        start_time = time.time()  # Record start time
        
        feature = np.array(feature)
        feature = np.c_[np.ones((feature.shape[0], 1)), feature]  # Add bias term
        target = np.array(target)
        n_samples = feature.shape[0]
        
        # Split data into train and validation sets
        n_val = int(0.1 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        feature_train, target_train = feature[train_indices], target[train_indices]
        feature_val, target_val = feature[val_indices], target[val_indices]

        if method == 'gradient_descent':
            self.weights = np.random.randn(feature.shape[1], target.shape[1]) * 0.01
            if gradient_method == 's':
                batch_size = 1
            elif gradient_method == 'b':
                batch_size = n_samples
            elif gradient_method != 'm':
                raise ValueError("gradient_method must be one of 'stochastic':'s', 'batch':'b', or 'mini-batch':'m'")

            self.weights = self.gradient_descent(
                feature_train, target_train, feature_val, target_val, self.weights, feature_train.shape[0],
                gradient_method, batch_size, learning_rate, max_epochs, regularization, patience
            )
            
            self.best_weights = self.weights.copy()

        elif method == 'normal_equation':
            gradient_method = 'b' 
            batch_size = n_samples
            print("Running on Batch mode with Normal equation method") 
            self.weights = self.normal_equation(feature_train, target_train, regularization)
            self.best_weights = self.weights.copy()
        else:
            raise ValueError("method must be one of 'gradient_descent' or 'normal_equation'")
        
        end_time = time.time()  # Record end time
        training_time = end_time - start_time  # Compute time taken
        
        return self.best_weights, training_time
    
    def gradient_descent(self, X, y, feature_val, target_val, W, n_samples, approach, batch_size,
                        learning_rate=0.01, max_epochs=1000, reg_lambda=0, patience=10, tol=1e-6):
        """Solves Linear Regression using Gradient Descent with optional L2 Regularization and Early Stopping."""

        patience_counter = 0  # Initialize patience counter

        for epoch in range(max_epochs):
            
            if approach == 's':  # Stochastic Gradient Descent
                for i in range(n_samples):
                    xi = X[i:i+1]
                    yi = y[i:i+1]
                    y_pred = xi @ W
                    error = y_pred - yi
                    dW = (xi.T @ error) + reg_lambda * W
                    W -= learning_rate * dW                    
                    
                    val_loss = np.mean((feature_val @ W - target_val) ** 2)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_weights = W.copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch} with best val loss: {self.best_val_loss:.4f}")
                            W =  self.best_weights
                            return W  # Return the best weights found

            elif approach == 'm':  # Mini-batch Gradient Descent
                indices = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[indices], y[indices]
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    y_pred = X_batch @ W
                    error = y_pred - y_batch
                    dW = (X_batch.T @ error) / len(X_batch) + reg_lambda * W
                    W -= learning_rate * dW

                # Compute validation loss after each epoch
                val_loss = np.mean((feature_val @ W - target_val) ** 2)                    
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = W.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch} with best val loss: {self.best_val_loss:.4f}")
                        W =  self.best_weights 
                        return W # Return the best weights found
            else:  # Batch Gradient Descent
                y_pred = X @ W
                error = y_pred - y
                dW = (X.T @ error) / n_samples + reg_lambda * W
                W -= learning_rate * dW

            # Compute validation loss after each epoch
            val_loss = np.mean((feature_val @ W - target_val) ** 2)
            

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = W.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} with best val loss: {self.best_val_loss:.4f}")
                    W =  self.best_weights
                    return W  # Return the best weights found
            print(f"Method: {approach} ----, Epoch {epoch}, Val Loss: {val_loss:.4f}, Best Val Loss: {self.best_val_loss:.4f}")
            print(f"Best Weights After Epoch {epoch}: {self.best_weights.flatten()}")
            
            # Check convergence
            if np.linalg.norm(dW) < tol:
                break

        return W  # Return final weights after training
        
    def normal_equation(self, X, y, reg_lambda=0):
        """Solves Linear Regression using the Normal Equation with optional L2 Regularization."""
        n_features = X.shape[1]
        I = np.eye(n_features)
        W = inv(X.T @ X + reg_lambda * I) @ X.T @ y
        return W
        
    def predict(self, X):
        """Make predictions for given features."""
        if self.best_weights is None:
            raise ValueError("Model has not been trained or weights have not been loaded.")
            
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.best_weights
    
    def score(self, X, y):
        """Calculate Mean Squared Error between predictions and target values for multiple outputs."""
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
        """Save model parameters to a file."""
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        model_params = {
            'weights': self.best_weights,
        }
        
        try:
            np.savez(filepath, **model_params)
            print(f"Model parameters saved successfully to {filepath}")
        except Exception as e:
            raise Exception(f"Error saving model parameters: {str(e)}")
    
    def load(self, filepath):
        """Load model parameters from a file."""
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        try:
            loaded = np.load(filepath)
            self.best_weights = loaded['weights']
            print(f"Model parameters loaded successfully from {filepath}")
        except Exception as e:
            raise Exception(f"Error loading model parameters: {str(e)}")
