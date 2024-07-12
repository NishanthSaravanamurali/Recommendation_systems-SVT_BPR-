import numpy as np

# Function for SVT matrix completion with given dimensions and rank
def svt_matrix_completion(m, n, r, X, mask, tau=1.0, delta=1.0, max_iter=100, tol=1e-4):
    Y = np.zeros((m, n))
    for i in range(max_iter):
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        S_threshold = np.maximum(S - tau, 0)[:r]  # Only keep top r singular values
        X_hat = np.dot(U[:, :r], np.dot(np.diag(S_threshold), Vt[:r, :]))
        Y = Y + delta * (mask * (X - X_hat))
        if np.linalg.norm(mask * (X - X_hat), 'fro') < tol:
            break
    return X_hat

# Example usage with the provided conditions
m = 8  # Number of rows
n = 8  # Number of columns
r = 4  # Rank of the low-rank matrix

# Generate example data
X = np.random.randn(m, n)  # Example original matrix
mask = np.random.randint(0, 2, size=(m, n)).astype(bool)  # Example mask matrix

print("Original Matrix:")
print(X)

# Print rank of original matrix
print("Rank of Original Matrix:", np.linalg.matrix_rank(X))

# Perform SVT matrix completion
X_completed = svt_matrix_completion(m, n, r, X, mask)

print("Completed Low-Rank Matrix:")
print(X_completed)

# Print rank of completed matrix
print("Rank of Completed Matrix:", np.linalg.matrix_rank(X_completed))
