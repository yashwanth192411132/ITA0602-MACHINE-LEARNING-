
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Define sigmoid activation function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 5. Initialize network parameters
np.random.seed(42)
input_size = X_train.shape[1]
hidden_size = 16
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 1000
m = len(X_train)

# 6. Train using Backpropagation
for epoch in range(epochs):

    # Forward propagation
    Z1 = X_train @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # Binary cross-entropy loss
    eps = 1e-9
    loss = -np.mean(
        y_train * np.log(A2 + eps)
        + (1 - y_train) * np.log(1 - A2 + eps)
    )

    # Backward propagation
    dZ2 = A2 - y_train
    dW2 = (A1.T @ dZ2) / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = (X_train.T @ dZ1) / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# 7. Test the trained network
hidden_output = sigmoid(X_test @ W1 + b1)
output = sigmoid(hidden_output @ W2 + b2)

# Threshold: >= 0.5 means class 1 (benign)
y_pred = (output >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print("\nActual samples:", y_test[:10].ravel())
print("Predicted samples:", y_pred[:10].ravel())
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClass labels:")
print("0 = Malignant")
print("1 = Benign")
