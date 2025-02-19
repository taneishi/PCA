import numpy as np
import matplotlib.pyplot as plt

def mean_centering(X):
    N, n = X.shape

    mu = sum(X[i, :] for i in range(N)) / N

    print('Original mean', mu.round(3))

    assert np.allclose(mu, np.mean(X, axis=0))

    X = X - mu

    print('Centered by the mean', np.mean(X, axis=0).round(3))

    return X

def covariance_matrix(X):
    N, n = X.shape

    # X is centered by the mean.
    assert np.allclose(np.mean(X, axis=0), np.zeros(n))

    cov = np.zeros((n, n))

    # By definition.
    for i, X_i in enumerate(X.T):
        for j, X_j in enumerate(X.T):
            cov[i, j] = np.sum(X_i.A * X_j.A) / (N - 1)

    # Simplified way.
    C = X.T * X / (N - 1)

    assert np.allclose(C, cov)
    assert np.allclose(C, np.cov(X.T))

    print('Variances for each dimension {:.3f} {:.3f}'.format(np.var(X[:, 0], ddof=1), np.var(X[:, 1], ddof=1)))

    assert np.isclose(np.var(X[:, 0], ddof=1), C[0, 0])
    assert np.isclose(np.var(X[:, 1], ddof=1), C[1, 1])

    print('Covariance matrix\n', C.round(3))

    return C

def pca_eig(X, C):
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Sort eigenvalues in the descending order.
    indices = np.flip(np.argsort(eigenvalues))
    L = np.diag(eigenvalues[indices])
    W = np.asmatrix(eigenvectors[:, indices])

    print('Eigenvalues\n', L.round(3))
    print('Eigenvectors\n', W.round(3))

    # C = W L W.T
    assert np.allclose(C, W * L * W.T)

    # W^{-1} = W.T
    assert np.allclose(W * W.T, np.identity(len(eigenvalues)))

    # Projection.
    T = X * W

    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    plt.scatter(X[:, 0].A, X[:, 1].A, marker='+', linewidth=0.8, color='black')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.axline((0, 0), (1, 0), linestyle='--', linewidth=0.5, color='black')
    plt.axline((0, 0), (0, 1), linestyle='--', linewidth=0.5, color='black')
    plt.axline((0, 0), (W.T[0, 0], W.T[0, 1]), linestyle='--', linewidth=0.8, color='black')
    plt.axline((0, 0), (W.T[1, 0], W.T[1, 1]), linestyle='--', linewidth=0.8, color='black')
    plt.annotate('$X_1$', xy=(3.5, 0.1))
    plt.annotate('$X_2$', xy=(0.05, 3.6))
    plt.annotate('$Y_1$', xy=(3.5, -1.8))
    plt.annotate('$Y_2$', xy=(2.2, 3.6))
    plt.savefig('figure/eigenvectors.png', dpi=100)

    return T

def pca_svd(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    print('Singular values', S.round(3))

    # X = U S V.T
    assert np.allclose(X, U * np.diag(S) * Vt)

    # X V = U S
    assert np.allclose(X * Vt.T, U * np.diag(S))

    # Projection.
    T = X * Vt.T

    return T

def main():
    # Pearson' dataset.
    X = np.asmatrix([
        [0.0, 5.9],
        [0.9, 5.4],
        [1.8, 4.4],
        [2.6, 4.6],
        [3.3, 3.5],
        [4.4, 3.7],
        [5.2, 2.8],
        [6.1, 2.8],
        [6.5, 2.4],
        [7.4, 1.5]
        ])

    X = mean_centering(X)

    C = covariance_matrix(X)

    T_eig = pca_eig(X, C)

    T_svd = pca_svd(X)

    # Projected coodinates are the same except for their signs.
    T_svd[:, 1] = -T_svd[:, 1]

    assert np.allclose(T_eig, T_svd)

    N, n = X.shape

    L, W = np.linalg.eig(X.T * X / (N-1))
    U, S, Vt = np.linalg.svd(X)

    # L = cov(T.T)
    assert np.allclose(np.diag(L), np.cov(T_eig.T))

    # L = S^2 / (N - 1)
    assert np.allclose(L, S ** 2 / (N - 1))

    # U * U.T = I
    assert np.allclose(U * U.T, np.diag(np.ones(U.shape[0])))

    # Vt * Vt.T = I
    assert np.allclose(np.diag(np.ones(Vt.shape[0])), Vt * Vt.T)

    assert np.allclose(L / np.sum(L), (S**2) / np.sum(S**2))

    print('The ratio of the variance for each projected coodinate', np.round(L / np.sum(L), 3))

if __name__ == '__main__':
    main()
