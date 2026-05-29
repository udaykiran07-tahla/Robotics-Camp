# Use small subset for faster training
X_train = X_train[:5000].reshape(-1, 784) / 255.0
y_train = y_train[:5000]

X_test = X_test.reshape(-1, 784) / 255.0

# One-hot encoding
Y = np.eye(10)[y_train]

# Network parameters
W1 = np.random.randn(784, 64) * 0.01
b1 = np.zeros((1, 64))

W2 = np.random.randn(64, 10) * 0.01
b2 = np.zeros((1, 10))

# Functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

losses = []
lr = 0.1

# Training
for epoch in range(5):

    # Forward
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Loss
    loss = -np.mean(np.sum(Y * np.log(A2 + 1e-8), axis=1))
    losses.append(loss)

    # Backprop
    m = X_train.shape[0]

    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1

    W2 -= lr * dW2
    b2 -= lr * db2

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

# Accuracy
A1 = relu(X_test @ W1 + b1)
A2 = softmax(A1 @ W2 + b2)

pred = np.argmax(A2, axis=1)

acc = np.mean(pred == y_test) * 100

print(f"\nTest Accuracy: {acc:.2f}%")

# Loss graph
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.show()