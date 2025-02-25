import numpy as np
from numpy.linalg import inv
import os

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.best_weights = None
        self.best_val_loss = float('inf')
        self.classes_ = None
    
    def sigmoid(self, z):
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to avoid overflow
    
    def softmax(self, z):
        """Compute softmax probabilities for multiclass classification."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, feature, target, method='gradient_descent', gradient_method='m', 
            batch_size=32, learning_rate=0.01, regularization=0, max_epochs=100, patience=3):
        feature = np.array(feature)
        target = np.array(target)
        
        self.classes_ = np.unique(target)
        n_classes = len(self.classes_)
        
        if n_classes > 2:
            y_one_hot = np.zeros((target.shape[0], n_classes))
            for i, c in enumerate(self.classes_):
                y_one_hot[:, i] = (target == c).astype(int)
            target = y_one_hot
        else:
            target = target.reshape(-1, 1)
        
        feature = np.c_[np.ones((feature.shape[0], 1)), feature]
        n_samples, n_features = feature.shape
        
        n_val = int(0.1 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        feature_train, target_train = feature[train_indices], target[train_indices]
        feature_val, target_val = feature[val_indices], target[val_indices]
        
        if method == 'gradient_descent':
            if n_classes <= 2:
                self.weights = np.random.randn(n_features, 1) * 0.01
            else:
                self.weights = np.random.randn(n_features, n_classes) * 0.01
            
            if gradient_method == 's':
                batch_size = 1
            elif gradient_method == 'b':
                batch_size = n_samples
            elif gradient_method != 'm':
                raise ValueError("gradient_method must be 's', 'b', or 'm'")
            
            self.weights = self.gradient_descent(
                feature_train, target_train, feature_val, target_val, self.weights, feature_train.shape[0],
                gradient_method, batch_size, learning_rate, max_epochs, regularization, patience
            )
            self.best_weights = self.weights.copy()
        elif method == 'normal_equation':
            print("Running Normal Equation (IRLS) for Logistic Regression")
            self.weights = self.normal_equation(feature, target, regularization, max_epochs)
            self.best_weights = self.weights.copy()
        
        else:
            raise ValueError("method must be 'gradient_descent' or 'normal_equation'")

        return  self
    
    def log_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if y_true.shape[1] > 1:
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def gradient_descent(self, X, y, feature_val, target_val, W, n_samples, approach, batch_size,
                        learning_rate=0.01, max_epochs=1000, reg_lambda=0, patience=3):
        patience_counter = 0
        
        for epoch in range(max_epochs):
            if approach == 's':
                for i in range(n_samples):
                    xi = X[i:i+1]
                    yi = y[i:i+1]
                    z = xi @ W
                    y_pred = self.softmax(z) if y.shape[1] > 1 else self.sigmoid(z)
                    error = y_pred - yi
                    dW = (xi.T @ error) + reg_lambda * W
                    W -= learning_rate * dW
            else:
                indices = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[indices], y[indices]
                for i in range(0, n_samples, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    z = X_batch @ W
                    y_pred = self.softmax(z) if y.shape[1] > 1 else self.sigmoid(z)
                    error = y_pred - y_batch
                    dW = (X_batch.T @ error) / len(X_batch) + reg_lambda * W
                    W -= learning_rate * dW
            
            val_probs = self.softmax(feature_val @ W) if target_val.shape[1] > 1 else self.sigmoid(feature_val @ W)
            val_loss = self.log_loss(target_val, val_probs)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = W.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    return self.best_weights
        
        return W
        
    def normal_equation(self, X, y, reg_lambda=0, max_iter=100, tol=1e-6):
        """
        Solve logistic regression using the iteratively reweighted least squares (IRLS) method.
        This is the "normal equation" equivalent for logistic regression.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_classes)
            Target values.
        reg_lambda : float, default=0
            The regularization strength.
        max_iter : int, default=100
            Maximum number of iterations.
        tol : float, default=1e-6
            Tolerance for convergence.
            
        Returns:
        --------
        W : array-like
            The optimized weights.
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        if y.shape[1] > 1:  # Multi-class
            n_classes = y.shape[1]
            W = np.zeros((n_features, n_classes))
            
            # For multi-class, solve each class as a one-vs-rest problem
            for i in range(n_classes):
                W[:, i] = self._irls_binary(X, y[:, i:i+1], reg_lambda, max_iter, tol).flatten()
        else:  # Binary
            W = self._irls_binary(X, y, reg_lambda, max_iter, tol)
            
        return W
    
    def _irls_binary(self, X, y, reg_lambda, max_iter, tol):
        """Helper method for binary IRLS."""
        n_samples, n_features = X.shape
        W = np.zeros((n_features, 1))
        I = np.eye(n_features)
        
        for i in range(max_iter):
            z = X @ W
            p = self.sigmoid(z)
            
            # Diagonal weight matrix
            W_diag = p * (1 - p)
            W_mat = np.diag(W_diag.flatten())
            
            # Working response
            z_adj = z + (y - p) / (W_diag + 1e-10)
            
            # Update weights
            W_new = inv(X.T @ W_mat @ X + reg_lambda * I) @ X.T @ W_mat @ z_adj
            
            # Check convergence
            if np.linalg.norm(W - W_new) < tol:
                W = W_new
                break
                
            W = W_new
            
        return W
    
    def score(self, X, y):
        """Compute accuracy of the model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    def predict_proba(self, X):
        if self.best_weights is None:
            raise ValueError("Model has not been trained or weights have not been loaded.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        z = X @ self.best_weights
        return self.softmax(z) if self.best_weights.shape[1] > 1 else self.sigmoid(z)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    
    def save(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez(filepath, best_weights=self.best_weights, best_val_loss=self.best_val_loss, classes=self.classes_)
    
    def load(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        loaded = np.load(filepath, allow_pickle=True)
        self.best_weights = loaded['best_weights']
        self.best_val_loss = float(loaded['best_val_loss'])
        self.classes_ = loaded['classes']


