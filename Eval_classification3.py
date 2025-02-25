from Logistic_Regression import LogisticRegression
import numpy as np
from train_classification_3 import X_test,y_test,target_names
from sklearn.metrics import classification_report
# Load all model

mini_batch_model = LogisticRegression()
mini_batch_model.load("classification_mini_model_3.npz")

m_loaded_acc = mini_batch_model.score(X_test, y_test)
print(f"Mini Batch Loaded Model Accuracy: {m_loaded_acc:.4f}")

batch_model = LogisticRegression()
batch_model.load("classification_batch_model_3.npz")

b_loaded_acc = batch_model.score(X_test, y_test)
print(f"Batch Loaded Model Accuracy: {b_loaded_acc:.4f}")

stochastic_model = LogisticRegression()
stochastic_model.load("classification_stochastic_model_3.npz")

s_loaded_acc = stochastic_model.score(X_test, y_test)
print(f"stochastics Loaded Model Accuracy: {s_loaded_acc:.4f}")

normal_equation_model = LogisticRegression()
normal_equation_model.load("classification_normal_model_3.npz")

n_loaded_acc = normal_equation_model.score(X_test, y_test)
print(f"Loaded Model Accuracy: {n_loaded_acc:.4f}")




## best model Prediction
best = [mini_batch_model, batch_model, stochastic_model, normal_equation_model][np.argmax([m_loaded_acc, b_loaded_acc, s_loaded_acc, n_loaded_acc])]
print(best)
best.save("best_classification_model_3.npz")


best_model = LogisticRegression()
best_model.load("best_classification_model_3.npz")

y_pred = best_model.predict(X_test)
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