import numpy as np
import sklearn.datasets

# For verificatoin.
import sklearn.decomposition

class PCA:
    def __init__(self, method='svd', n_components=2):
        self.n_components = n_components
        self.method = method

    def svd(self):
        # Singular value decomposition
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)

        # Singular values are already sorted.
        assert np.allclose(S, np.flip(np.sort(S)))

        assert np.allclose(self.X, U * np.diag(S) * Vt)

        # T = U * S
        pcUdS = U[:, :self.n_components] * np.diag(S[:self.n_components])

        # T = X * V
        self.V = Vt.T
        pcXW = self.X * self.V[:, :self.n_components]

        # X * V = U * S
        assert np.allclose(pcXW, pcUdS)

        return pcUdS

    def eig(self):
        # Convariance matrix
        C = self.X.T * self.X / (self.N - 1)
        assert np.allclose(C, np.cov(self.X.T))

        # Eigendecompostion of the covariance matrix.
        L, W = np.linalg.eig(C)

        # Sort eigenvectors in the descending order of eigenvalues.
        self.W = W[:, np.flip(np.argsort(L))]
        L = np.flip(np.sort(L))

        assert np.allclose(C, self.W * np.diag(L) * self.W.T)
        assert np.allclose(self.W * self.W.T, np.identity(len(L)))

        # T = X * W
        return self.X * self.W[:, :self.n_components]

    def restore(self, X):
        if self.method == 'svd':
            return X * self.V.T + self.mu
        elif self.method == 'eig':
            return X * self.W.T + self.mu

    def fit_transform(self, X):
        self.N, self.n = X.shape
        self.mu = np.mean(X, axis=0)

        # Mean centering.
        X = X - self.mu

        self.X = np.asmatrix(X)
        
        if self.method == 'svd':
            transformed = self.svd()
        elif self.method == 'eig':
            transformed = self.eig()

        return transformed

def main(n_components=2):
    # Fisher iris dataset.
    data = sklearn.datasets.load_iris()
    X = data['data']

    pca = PCA(n_components=n_components, method='eig')
    eig_pc = pca.fit_transform(X)
    
    pca = PCA(n_components=n_components, method='svd')
    svd_pc = pca.fit_transform(X)

    pca = sklearn.decomposition.PCA(n_components=n_components)
    sklearn_pc = pca.fit_transform(X)

    # Principal components are the same except for the signs.
    sklearn_pc[:, 1] = -sklearn_pc[:, 1]

    # Assert that SVD and Eig results the same.
    assert np.allclose(eig_pc, svd_pc)
    assert np.allclose(sklearn_pc, svd_pc)

    pca = PCA(n_components=X.shape[1], method='svd')
    svd_pc = pca.fit_transform(X)

    # Verify the restoration.
    assert np.allclose(X, pca.restore(svd_pc))

if __name__ == '__main__':
    main()
