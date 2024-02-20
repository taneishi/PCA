import numpy as np

class PCA:
    def __init__(self, n_components=2, method='svd'):
        self.n_components = n_components
        self.method = method

    def svd(self):
        # Singular value decomposition
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)

        # Singular values are already sorted.
        assert np.allclose(S, np.flip(np.sort(S)))

        # Sort vectors in the descending order of singular values.
        #Vt = Vt[:, np.flip(np.argsort(S))]
        #S = np.flip(np.sort(S))

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

        # Eigendecomposition of the covariance matrix C.
        # L are eigenvalues and W are eigenvectors.
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
        
        # Mean centering.
        self.mu = np.mean(X, axis=0)
        X = X - self.mu
        
        self.X = np.asmatrix(X)
        
        if self.method == 'svd':
            transformed = self.svd()
        elif self.method == 'eig':
            transformed = self.eig()

        return transformed
