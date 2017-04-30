from numpy import exp, log
from functools import partial
from scipy.special import gamma, gammaln

def bhc(dat, family, alpha, r = 0.001):
    """Return a matrix in the format of linkage matrix for dendrogram
        @dat: N records of data with k columns
        @family: function to specify distribution for data. {"multivariate", "bernoulli"}
        @alpha: hyperparameter for the prior
        @r: scaling factor on the prior precision of the mean
    """
    N, k = dat.shape

    if family == "multivariate":
        m = np.mean(dat, axis=0).reshape(k, 1)
        S = np.cov(dat.T)/10 # precision?
        mlfunc = partial(niw, m=m, S=S, r=r)
    elif family == "bernoulli":
        cc=0.01
        m = np.mean(np.vstack((dat, np.ones(k)*cc, np.zeros(k))), axis=0)
        alp= m*2; beta=(1-m)*2
        mlfunc = partial(bb, α=alp, β=beta)

    # leaf nodes
    SS = list(range(N))
    x0 = []; d0 = [alpha] * N
    ml = []
    for l in range(N):
        x0.append((l,))
        ml.append(mlfunc(dat[l,].reshape(1,k)))

    # paired base cases
    t = 0; PP = []
    c1 = []; c2 = []
    x = []; d = []
    lp1 = []; lp2 = []; lodds = []
    for i in range(N-1):
        for j in range(i+1, N):
            c1.append(i); c2.append(j)
            x.append(x0[i]+x0[j])
            d.append((alpha * gamma(len(x[t])) + d0[i] * d0[j]))
            lp1.append(mlfunc(dat[x[t],:]) + np.log(alpha) + gammaln(len(x[t])) - np.log(d[t]))
            lp2.append(ml[i] + ml[j] + np.log(d0[i]) + np.log(d0[j]) - np.log(d[t]))
            lodds.append(lp1[t] - lp2[t])
            PP.append(t); t = t + 1
        # build tree, Z = [leaf1, leaf2, weight, #leaves]
    p = 0
    Z = []
    dye = {}
    while(1):
        idx = lodds.index(max([lodds[y] for y in PP]))
        Z.append([c1[idx], c2[idx], 1/lodds[idx], len(x[idx])])
        if lodds[idx] < 0:
            dye[N + p] = "#FF0000"
        else:
            dye[N + p] = "#0013FF"

        x0.append(x[idx]); d0.append(d[idx]); ml.append(log(exp(lp1[idx])+exp(lp2[idx])))
        rm = set(Z[p][:2])
        SS = [y for y in SS if y not in rm]
        if len(SS) == 0:
            break

        for q in SS:
            c1.append(N+p); c2.append(q)
            x.append(x0[N+p] + x0[q])
            d.append(alpha * gamma(len(x[t])) + d0[N+p] * d0[q])

            lp1.append(mlfunc(dat[x[t],:]) + np.log(alpha) + gammaln(len(x[t])) - np.log(d[t]))
            lp2.append(ml[N+p] + ml[q] + np.log(d0[N+p]) + np.log(d0[q]) - np.log(d[t]))
            lodds.append(lp1[t] - lp2[t])
            PP.append(t); t = t + 1

        PP = [y for y in PP if c1[y] not in rm and c2[y] not in rm]
        SS.append(N + p); p = p + 1

    Z_ = weighted(Z, N)

    return Z_, dye


def weighted(Z, N):
    mw = max([y[2] for y in Z])
    for i in range(len(Z)):
        if Z[i][2] < 0:
            Z[i][2] = 2 * mw
        if Z[i][0] > (N - 1):
            Z[i][2] += Z[Z[i][0] - N][2]
        if Z[i][1] > (N - 1):
            Z[i][2] += Z[Z[i][1] - N][2]
    return Z


def scale_matrix(X, N, k, r, m, S):
    """Return scale matrix for the inverse-Wishart distribution on Sigma.
        @X: N records of data with k columns
        @m: prior on the mean, k * 1
        @S: prior on the covariance, k * k
    """

    xsum = np.sum(X, axis = 0).reshape(k,1) # column sum
    t1 = X.T @ X
    t2 = r * N / (N + r) * (m @ m.T)
    t3 = 1/(N+r) * (xsum @ xsum.T)
    t4 = (r / (N + r)) * (m @ xsum.T + xsum @ m.T)

    Sprime = S + t1 + t2 - t3 - t4  #? +t3

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
    Sprime = scale_matrix(X, N, k, r, m, S)

    t1 = (2 * np.pi) ** (- N * k / 2)
    t2 = (r / (N + r)) ** (k/2)
    t3 = np.linalg.det(S) ** (v/2)
    t4 = np.linalg.det(Sprime) ** (-vprime/2)
    t5num = np.prod(gamma( (vprime - np.arange(k))/2 ) ) * (2 ** (vprime * k / 2))
    t5den = np.prod(gamma( (v - np.arange(k))/2 ) ) * (2 ** (v * k / 2))

    ml = t1 * t2 * t3 * t4 * (t5num/t5den)

    return np.log(ml)

def bb(X, α=0.001, β=0.01):
    """Return marginal likelihood for bernoulli data using the conjugate prior distribution Bernoulli-Beta
       @X: N records of data with k columns
       @α, β: hyperparmeter for Beta distribution
    """
    md = np.sum(X,axis=0)
    N = X.shape[0]
    num = gammaln(α+β) + gammaln(α+md) + gammaln(β+N-md)
    den = gammaln(α) + gammaln(β) + gammaln(α+β+N)
    return np.sum(num - den)
