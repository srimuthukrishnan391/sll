# ============================================================
# SIMPLE SELF LEARNING ALGORITHM (ASLA) DEMONSTRATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------
# 1. CREATE DEMONSTRATION DATA
# ------------------------------------------------------------

n_samples = 300

# Two simple classes
X_normal = np.random.normal(loc=0.0, scale=1.0, size=(150, 2))
X_abnormal = np.random.normal(loc=2.5, scale=1.0, size=(150, 2))

X = np.vstack([X_normal, X_abnormal])
y = np.hstack([
    np.zeros(150),       # Normal = 0
    np.ones(150)         # Abnormal = 1
])

# Shuffle
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]

# ------------------------------------------------------------
# 2. SIMPLE TWO LAYER MODEL
# ------------------------------------------------------------

input_dim = 2
hidden_dim = 8

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, 1) * 0.1
b2 = np.zeros(1)

# ------------------------------------------------------------
# 3. LEARNING PARAMETERS
# ------------------------------------------------------------

learning_rate = 0.05
momentum = 0.9

vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)

vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)

# ------------------------------------------------------------
# 4. ACTIVATION FUNCTIONS
# ------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def tanh(x):
    return np.tanh(x)


# ------------------------------------------------------------
# 5. TRAINING
# ------------------------------------------------------------

loss_history = []

for epoch in range(100):

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    hidden = tanh(X @ W1 + b1)

    logits = hidden @ W2 + b2

    predictions = sigmoid(logits).ravel()

    # --------------------------------------------------------
    # Calculate error
    # --------------------------------------------------------

    error = predictions - y

    # --------------------------------------------------------
    # SELF LEARNING PART
    # --------------------------------------------------------
    # Difficult samples have larger error.
    # Give difficult samples higher learning weight.

    difficulty = np.abs(error)

    sample_weights = 1.0 + 2.0 * difficulty

    # Normalize weights
    sample_weights = sample_weights / np.mean(sample_weights)

    # --------------------------------------------------------
    # Weighted loss
    # --------------------------------------------------------

    eps = 1e-8

    loss = -np.mean(
        sample_weights *
        (
            y * np.log(predictions + eps)
            +
            (1 - y) * np.log(1 - predictions + eps)
        )
    )

    loss_history.append(loss)

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    weighted_error = error * sample_weights

    dW2 = hidden.T @ weighted_error[:, None] / len(X)
    db2 = np.mean(weighted_error)

    dhidden = weighted_error[:, None] @ W2.T

    dz1 = dhidden * (1 - hidden ** 2)

    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0)

    # --------------------------------------------------------
    # MOMENTUM UPDATE
    # --------------------------------------------------------

    vW2 = momentum * vW2 - learning_rate * dW2
    vb2 = momentum * vb2 - learning_rate * db2

    vW1 = momentum * vW1 - learning_rate * dW1
    vb1 = momentum * vb1 - learning_rate * db1

    W2 += vW2
    b2 += vb2

    W1 += vW1
    b1 += vb1

    # --------------------------------------------------------
    # ADAPTIVE LEARNING RATE
    # --------------------------------------------------------

    if epoch > 0:

        if loss_history[-1] < loss_history[-2]:
            learning_rate *= 1.01
        else:
            learning_rate *= 0.7

        learning_rate = np.clip(
            learning_rate,
            0.001,
            0.2
        )

    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if (epoch + 1) % 10 == 0:

        predicted_class = (predictions >= 0.5).astype(int)

        accuracy = np.mean(predicted_class == y)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy*100:.2f}% | "
            f"LR: {learning_rate:.5f}"
        )


# ------------------------------------------------------------
# 6. FINAL PREDICTION
# ------------------------------------------------------------

hidden = tanh(X @ W1 + b1)

probabilities = sigmoid(hidden @ W2 + b2).ravel()

final_prediction = (probabilities >= 0.5).astype(int)

accuracy = np.mean(final_prediction == y)

print("\n===================================")
print("SELF LEARNING DEMONSTRATION")
print("===================================")
print(f"Final Accuracy : {accuracy*100:.2f}%")
print(f"Final Learning Rate : {learning_rate:.5f}")


# ------------------------------------------------------------
# 7. SHOW DIFFICULT SAMPLES
# ------------------------------------------------------------

difficulty = np.abs(probabilities - y)

hard_indices = np.argsort(difficulty)[-10:]

print("\nMost difficult samples:")

for i in hard_indices:
    print(
        f"Sample {i:3d} | "
        f"True = {int(y[i])} | "
        f"Prediction = {probabilities[i]:.3f} | "
        f"Difficulty = {difficulty[i]:.3f}"
    )


# ------------------------------------------------------------
# 8. LOSS GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(7, 4))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Self Learning Training")

plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 9. DATA VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Normal"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Abnormal"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Demonstration Dataset")

plt.legend()
plt.grid(True)
plt.show()
