from Logistic_Regression import LogisticRegression
import numpy as np
from train_classification_3 import X_test,y_test,target_names
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
# Load all model
def plot_decision_boundary(model, X, y, filename):
    """
    Plots and saves the decision boundary for a trained logistic regression model.
    
    Parameters:
    -----------
    model : LogisticRegression
        The trained logistic regression model.
    X : array-like of shape (n_samples, 2)
        The feature matrix (only two features should be used for visualization).
    y : array-like of shape (n_samples,)
        The target labels.
    filename : str
        The name of the file to save the plot.
    """
    
    if X.shape[1] != 2:
        raise ValueError("This function only supports plotting for 2D feature space.")
    
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X=X, y=y, clf=model, legend=2)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    
    # Save plot
    plt.savefig(f"plots/{filename}.png")
    plt.close()
    
    print(f"Plot saved as 'plots/{filename}.png'")

mini_batch_model = LogisticRegression()
mini_batch_model.load("models/classification_mini_model_3.npz")
y_pred = mini_batch_model.predict(X_test)
plot_decision_boundary(mini_batch_model,X_test,y_pred,"Plot-logistic-regression-mini-model-3")

m_loaded_acc = mini_batch_model.score(X_test, y_test)
print(f"Mini Batch Loaded Model Accuracy: {m_loaded_acc:.4f}")

batch_model = LogisticRegression()
batch_model.load("models/classification_batch_model_3.npz")
y_pred = batch_model.predict(X_test)
plot_decision_boundary(batch_model,X_test,y_pred,"Plot-logistic-regression-batch-model-3")

b_loaded_acc = batch_model.score(X_test, y_test)
print(f"Batch Loaded Model Accuracy: {b_loaded_acc:.4f}")

stochastic_model = LogisticRegression()
stochastic_model.load("models/classification_stochastic_model_3.npz")
y_pred = stochastic_model.predict(X_test)
plot_decision_boundary(stochastic_model,X_test,y_pred,"Plot-logistic-regression-stochastic-model-3")

s_loaded_acc = stochastic_model.score(X_test, y_test)
print(f"stochastics Loaded Model Accuracy: {s_loaded_acc:.4f}")

normal_equation_model = LogisticRegression()
normal_equation_model.load("models/classification_normal_model_3.npz")
y_pred = normal_equation_model.predict(X_test)
plot_decision_boundary(normal_equation_model,X_test,y_pred,"Plot-logistic-regression-normal-model-3")

n_loaded_acc = normal_equation_model.score(X_test, y_test)
print(f"Loaded Model Accuracy: {n_loaded_acc:.4f}")




## best model Prediction
best = [mini_batch_model, batch_model, stochastic_model, normal_equation_model][np.argmax([m_loaded_acc, b_loaded_acc, s_loaded_acc, n_loaded_acc])]
print(best)
best.save("models/best_classification_model_3.npz")


best_model = LogisticRegression()
best_model.load("models/best_classification_model_3.npz")

y_pred = best_model.predict(X_test)
plot_decision_boundary(best_model,X_test,y_pred,"Plot-logistic-regression-best-model-3")
print("\nClassification Report:")
print(np.unique(y_test))
print(classification_report(y_test, y_pred, target_names=target_names))

# Print a few predictions
print("\nSample Predictions:")
for i in range(5):
    probs = best_model.predict_proba(X_test[i:i+1])[0]
    pred_class = best_model.predict(X_test[i:i+1])[0]
    actual_class = y_test[i]
    
    print(f"Sample {i+1}:")
    print(f"  Probabilities: {', '.join([f'{target_names[j]}: {p:.4f}' for j, p in enumerate(probs)])}")
    print(f"  Predicted: {target_names[pred_class]}")
    print(f"  Actual: {target_names[actual_class]}")
    print()