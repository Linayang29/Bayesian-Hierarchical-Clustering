import numpy as np
from scipy.special import gamma

def bhc(dat, family, alpha, r):
    """Return a matrix in the format of linkage matrix for dendrogram
        @dat: N records of data with k columns
        @family: function to specify prior distribution. Only 'niw' for now
        @alpha: hyperparameter for the prior
        @r: scaling factor on the prior precision of the mean
    """
    N, k = dat.shape
    m = np.mean(dat, axis=0).reshape(k, 1)
    S = np.cov(dat.T)  # precision?

    # labels

    # build tree

    return Z


def scale_matrix(X, m, S):
    """Return scale matrix for the inverse-Wishart distribution on Sigma.
        @X: N records of data with k columns
        @m: prior on the mean, k * 1
        @S: prior on the covariance, k * k
    """

    xsum = np.sum(X, axis = 0).reshape(d,1) # column sum
    t1 = X.T @ X
    t2 = r * n / (n + r) * (m @ m.T)
    t3 = 1/(n+r) * (xsum @ xsum.T)
    t4 = (r / (n + r)) * (m @ xsum.T + xsum @ m.T)

    Sprime = S + t1 + t2 + t3 - t4

    return Sprime


def niw(X, m, S, r):
    """Return marginal likelihood for multivariate normal data using the conjugate prior distribution normal-inverse-Wishart
       @X: N records of data with k columns
       @m: prior on the mean, k * 1
       @S: prior on the covariance, k * k
       @r: scaling factor on the prior precision of the mean
    """

    N, k = X.shape
    v = k
    vprime = v + N
    Sprime = scale_matrix(S)

    t1 = (2 * np.pi) ** (- N * k / 2)
    t2 = (r / (N + r)) ** (k/2)
    t3 = np.linalg.det(S) ** (v/2)
    t4 = np.linalg.det(Sprime) ** (-vprime/2)
    t5num = np.prod(gamma( (vprime + 1 - np.arange(1, d+1))/2 ) ) * (2 ** (vprime * k / 2))
    t5den = np.prod(gamma( (v + 1 - np.arange(1, d+1))/2 ) ) * (2 ** (v * k / 2))

    ml = t1 + t2 + t3 + t4 + (t5num/t5den)

    return ml
