import numpy as np
from Utilities import obsFrac, scale, descale, alignSeries
from kalman import Kalman
from scipy.stats import linregress
from scipy.linalg import block_diag
from scipy.signal import detrend
from sklearn import linear_model

##### Functions for generating initial SSM parameter/hidden state estimates
def pcaEst(Y, nLF):
    """
    Performs PCA fit to Lambda, F_t assuming no missing data using the method from from Bai+Ng 2002. Assumes
    the data is already standardized/preprocessed.

    Arguments
    -Y: N x T numpy array
        Data array
    -nLF: int
        Number of latent variables to use

    Returns
    -N x T numpy array
        Estimate of Y from estimated factors and loadings
    -N x nLF numpy array
        Estimate of loadings
    -nLF x T numpy array
        Estimate of factors
    """
    N = Y.shape[0]

    # Observation covariance
    SigmaY = np.dot(Y, Y.T)
    # Find eigenvalues/vectors
    eVals, eVecs = np.linalg.eig(SigmaY)
    # Sort eigenvalues
    eValIdx = np.argsort(eVals)[::-1]

    # Get matrix of eigenvectors corresponding to nLF largest eigenvalues
    l = np.sqrt(N) * eVecs[:, eValIdx[0:nLF]]

    # Use PCA result to estimate factors
    fT = np.dot(l.T, Y) / float(N)

    return np.dot(l, fT), l, fT

def pcaEstMD(Y, nLF, maxIt=11):
    """
    Performs PCA fit to Lambda, F_t using the missing data strategy from Bai+Ng 2002. Assumes the data is
    already standardized/preprocessed.

    Arguments
    -Y: N x T numpy array
        Data array
    -nLF: int
        Number of latent variables to use

    Returns
    -N x T numpy array
        Estimate of Y from estimated factors and loadings
    -N x nLF numpy array
        Estimate of loadings
    -nLF x T numpy array
        Estimate of factors
    """
    Ytmp = Y.copy()
    N = Ytmp.shape[0]

    # Replace nans with average to start
    rowNanIdx, colNanIdx = np.where(np.isnan(Y))
    Ytmp[(rowNanIdx, colNanIdx)] = np.nanmean(Y)

    l, fT = None, None

    for i in range(0, maxIt):
        # Get current lambda and F_t estimates
        _, l, fT = pcaEst(Ytmp, nLF)

        # Replace nans with new estimates
        for i, j in zip(rowNanIdx, colNanIdx):
            Ytmp[i, j] = np.dot(l[i, :], fT[:, j])

    return np.dot(l, fT), l, fT

def kfSetup(Y, C, fTHat, nLF, tVar=10):
    """
    Sets up matrices, initial state and initial covariance estimates for the Kalman filter

    Arguments:
    -Y: N x T float array
        Data matrix
    -C: N x nLF float array
        Loadings matrix
    -fTHat: nLF x T float array
        Hidden state estimate
    -nLF: int
        Number of latent factors
    -tVar: int
        Number of observations at beginning of series to use when computing hidden state noise estimate

    Returns
    -nLF numpy array
        Initial hidden state estimate (pi0)
    -nLF x nLF numpy array
        Initial hidden state covariance estimate (sigma0)
    -nLF x nLF
        Hidden state transition matrix (A)
    -nLF x nLF numpy array
        Hidden state noise estimate (Q)
    -N x N numpy array
        Observation noise estimate (R)
    """
    N, T = Y.shape
    # Estimate A by regressing state onto its lag
    clf = linear_model.LinearRegression()
    clf.fit(fTHat[:, 0:-1].T, fTHat[:, 1:].T)
    A = clf.coef_

    # Estimate state variance from fit tVar observations
    Q = np.diag(np.var(detrend(fTHat[:, 0:tVar]), axis=1))

    # Estimate observation variance. Initial guess shouldn't matter too much...
    obsVar = np.nanmean(np.nanvar(Y - np.dot(C, fTHat), axis=1, ddof=1))
    R = np.diag(N*[obsVar if obsVar != 0 else np.mean(Q[0:nLF, 0:nLF]) * 0.1])

    # Estimate observation variance
    #R = np.diag(np.nanvar(Y - np.dot(C, fTHat), axis=1, ddof=1))
    ## If there are no observations for a sensor, doesn't matter what gets filled in for its variance...
    #meanObsVar = np.nanmean(R)
    #R[np.isnan(R)] = meanObsVar

    pi0 = fTHat[:, 0].copy().T
    sigma0 = Q

    return pi0, sigma0, A, Q, R

##### Kalman smoother
def kalmanSmooth(Y, pi0, sigma0, A, C, Q, R, nLF):
    """
    Runs Kalman filtering and smoothing step of dynamic factor model estimator.

    Returns
    -nLF x T numpy array
        Smoothed means
    -T x nLF x nLF numpy array
        Smoothed covariances
    -T-1 x nLF x nLF numpy array
        Smoothed lagged covariances (ie, cov[x_t, x_t-1])
    """
    N, T = Y.shape

    # Initialize Kalman filter
    kf = Kalman(mu_0=pi0.copy(), sigma_0=sigma0.copy(), A=A, B=np.zeros(2*[2*nLF]), C=C, D=None, Q=Q, R=R)

    # sigma_t|t, mu_t|t
    sigma_filt = np.zeros([T, nLF, nLF])
    sigma_filt[0] = sigma0.copy()
    mu_filt = np.zeros([T, nLF])
    mu_filt[0] = pi0.copy()

    # sigma_t|t-1, mu_t|t-1. NOTE: indexed by t-1!!!
    sigma_pred = np.zeros([T-1, nLF, nLF])
    mu_pred = np.zeros([T-1, nLF])

    # Avoid printing repetitive errors
    printedPosSemidefErr = False

    # Filtering step
    for t in range(1, T):
        kf.predict()

        # Save mu_t|t-1 and sigma_t|t-1
        sigma_pred[t-1, :, :] = kf.sigma
        mu_pred[t-1] = kf.mu

        # Update if we have a measurement. Nans are handled by Kalman.
        kf.update(Y[:, t])

        # Save filtered mean, covariance
        sigma_filt[t, :, :] = kf.sigma
        mu_filt[t] = kf.mu

        # Make sure filtered covariance is positive semidefinite!
        eigs, _ = np.linalg.eig(sigma_filt[t])
        if len(np.where(eigs < 0)[0]) > 0 and not printedPosSemidefErr:
            print "\tsigma_filt[%i] is not positive semidefinite"%t
            printedPosSemidefErr = True

    # sigma_t|T, mu_t|T
    sigma_smooth = np.zeros((T, nLF, nLF))
    mu_smooth = np.zeros((T, nLF))

    # Initialize: sigma_T|T = sigma_T|T(filtered)
    sigma_smooth[-1] = sigma_filt[-1]
    # mu_T|T = mu_T|T(filtered)
    mu_smooth[-1] = mu_filt[-1]

    # Lagged covariance. Indexed by t-1.
    sigmaLag_smooth = np.zeros((T-1, nLF, nLF))
    # sigmaLag_{T,T-1} = (1 - K_T C) A V_{T-1|T-1}, where K_T is Kalman gain at last timestep.
    K_T = np.dot(sigma_pred[-1], np.dot(kf.C.T, np.linalg.inv(np.dot(kf.C, \
                                                                    np.dot(sigma_pred[-1], kf.C.T)) + kf.R)))
    sigmaLag_smooth[-1] = np.dot(np.dot((np.identity(nLF) - np.dot(K_T, kf.C)), kf.A), sigma_filt[-2])

    # Backwards Kalman gain
    J = np.zeros((T-1, nLF, nLF))

    # Smoothing step. Runs from t=T-1 to t=0.
    for t in range(T-2, -1, -1):
        # Backward Kalman gain matrix
        J[t] = np.dot(np.dot(sigma_filt[t], kf.A.T), np.linalg.inv(sigma_pred[t]))

        # Smoothed mean
        mu_smooth[t] = mu_filt[t] + np.dot(J[t], mu_smooth[t+1] - mu_pred[t])

        # Smoothed covariance
        sigma_smooth[t, :, :] = sigma_filt[t] + np.dot(np.dot(J[t], sigma_smooth[t+1] - sigma_pred[t]), J[t].T)

    # Lagged smoothed covariance. Pretty sure this is correct...
    for t in range(T-3, -1, -1):
        sigmaLag_smooth[t, :, :] = np.dot(sigma_filt[t+1], J[t].T) \
                    + np.dot(np.dot(J[t+1], (sigmaLag_smooth[t+1] - np.dot(kf.A, sigma_filt[t+1]))), J[t].T)

    # Fill in missing Y values
    YImp = Y.copy()
    nanRows, nanCols = np.where(np.isnan(YImp))

    for s, t in zip(nanRows, nanCols):
        YImp[s, t] = np.dot(C[s, :], mu_smooth[t, :])

    return mu_smooth.T, sigma_smooth, sigmaLag_smooth, YImp

##### M step of SSM EM algorithm
def mStep(YImp, XHat, P, PLag, A=None, C=None, Q=None, R=None):
    """
    Estimates SSM parameters given state and covariance estimates.

    Arguments
    -YImp: N x T
        Data matrix with nan values filled in by Kalman smoother
    -XHat: nLF x T
        Hidden state means E[x_t | y]
    -P: T x nLF x nLF
        Covariance estimate: E[x_t x_t^T | y]
    -PLag: T-1 x nLF x nLF
        Lagged covariance estimate: E[x_t x_{t-1}^T | y]
    -A, C, Q, R: providing a value for one of these fixes the parameter so it won't be estimated

    Returns
    -Estimates for A, C, Q, R, pi0 and sigma0
    """
    N, T = YImp.shape

    # Observation matrix
    CNew = np.dot(np.dot(YImp, XHat.T), np.linalg.inv(np.sum(P, axis=0))) if C is None else C

    # Observation noise covariance
    RNew = 1.0/float(T) * (np.dot(YImp, YImp.T) - np.dot(CNew, np.dot(XHat, YImp.T))) if R is None else R

    # State transition matrix
    ANew = np.dot(np.sum(PLag, axis=0), np.linalg.inv(np.sum(P[0:-1], axis=0))) if A is None else A

    # State noise covariance
    # TODO: CHECK THIS! Might need PLag.T...
    #QNew = 1.0/float(T-1) * (np.sum(P[1:], axis=0) - np.dot(ANew, np.sum(np.transpose(PLag, axes=(0, 2, 1)),
    #                                                                            axis=0))) if Q is None else Q
    QNew = 1.0/(T-1.0) * np.sum([p - np.dot(ANew, pl.T) for p, pl in zip(P[1:], PLag)], axis=0)

    # Initial state mean
    pi0New = XHat[:, 0]

    # Initial state covariance
    sigma0New = P[0, :, :] - np.outer(XHat[:, 0], XHat[:, 0].T)

    return ANew, CNew, QNew, RNew, pi0New, sigma0New

def ssmEM(Y, nLF, maxIt=50):
    """
    Runs state space EM algorithm

    Arguments
    -Y: N x T numpy array
        Data array
    -maxIt: int
        Number of iterations to run of EM

    Returns
    -XHat, sigma_smooth, A, C, Q, R, pi0, sigma0
    """
    N, T = Y.shape

    # Estimate SSM parameters with PCA, using EM to handle missing values
    _, C, fTHat = pcaEstMD(Y, nLF, maxIt=50)
    pi0, sigma0, A, Q, R = kfSetup(Y, C, fTHat, nLF)

    # Keep track of hidden state mean and covariance
    XHat = None
    sigma_smooth = None

    for i in range(maxIt):
        ##### E step
        # Estimate hidden state
        XHat, sigma_smooth, sigmaLag_smooth, YImp = kalmanSmooth(Y, pi0, sigma0, A, C, Q, R, nLF)

        # Second moment
        P = np.zeros((T, nLF, nLF))
        for t in range(T):
            P[t, :, :] = sigma_smooth[t, :, :] + np.outer(XHat[:, t], XHat[:, t].T)

        # Lagged second moment
        PLag = np.zeros((T-1, nLF, nLF))
        for t in range(T-1):
            PLag[t, :, :] = sigmaLag_smooth[t, :, :] + np.outer(XHat[:, t+1], XHat[:, t].T)

        ##### M step
        A, C, Q, R, pi0, sigma0 = mStep(YImp, XHat, P, PLag)

    # Finally, re-estimate hidden state    
    XHat, sigma_smooth, _, _ = kalmanSmooth(Y, pi0, sigma0, A, C, Q, R, nLF)

    return XHat, sigma_smooth, A, C, Q, R, pi0, sigma0

##### SSM EM with state augmented by velocity

def mStepAug(YImp, XHat, P, PLag, A):
    """
    Estimates SSM parameters given state and covariance estimates.

    Arguments
    -YImp: N x T
        Data matrix with nan values filled in by Kalman smoother
    -XHat: nLF x T
        Hidden state means E[x_t | y]
    -P: T x nLF x nLF
        Covariance estimate: E[x_t x_t^T | y]
    -PLag: T-1 x nLF x nLF
        Lagged covariance estimate: E[x_t x_{t-1}^T | y]
    -A, C, Q, R: providing a value for one of these fixes the parameter so it won't be estimated

    Returns
    -Estimates for A, C, Q, R, pi0 and sigma0
    """
    N, T = YImp.shape
    nLF = XHat.shape[0]/2

    # Observation matrix
    OmegaNew = np.dot(np.dot(YImp, XHat[0:nLF,:].T), np.linalg.inv(np.sum(P[:, 0:nLF, 0:nLF], axis=0)))

    # Observation noise covariance
    RNew = 1.0/float(T) * (np.dot(YImp, YImp.T) - np.dot(OmegaNew, np.dot(XHat[0:nLF, :], YImp.T)))

    # State noise covariance
    #QNew = 1.0/float(T-1) * (np.sum(P[1:], axis=0) - np.dot(A, np.sum(np.transpose(PLag, axes=(0, 2, 1)),
    #                                                                            axis=0)))
    QNew = 1.0/(T-1.0) * np.sum([p - np.dot(A, pl.T) - np.dot(pl, A.T) + np.dot(A, np.dot(p_prev, A.T))
                                                    for p, pl, p_prev in zip(P[1:], PLag, P[:-1])], axis=0)
    #print P[1]
    #print PLag[1]
    #print "Q eigenvalues: ", np.linalg.eig(QNew)[0]

    # Initial state mean
    pi0New = XHat[:, 0]

    # Initial state covariance
    sigma0New = P[0, :, :] - np.outer(XHat[:, 0], XHat[:, 0].T)

    return OmegaNew, QNew, RNew, pi0New, sigma0New

def pcaEstMDAug(Y, nLF, maxIt=10):
    """
    Performs PCA fit to Lambda, F_t using the missing data strategy from Bai+Ng 2002. Assumes the data is
    already standardized/preprocessed.
    """
    Ytmp = Y.copy()
    N = Ytmp.shape[0]

    # Replace nans with average to start
    rowNanIdx, colNanIdx = np.where(np.isnan(Y))
    Ytmp[(rowNanIdx, colNanIdx)] = np.nanmean(Y)

    OmegaHat, xHat = None, None

    for i in range(0, maxIt):
        # Get current lambda and F_t estimates
        _, OmegaHat, xHat = pcaEst(Ytmp, nLF)

        # Replace nans with new estimates
        for i, j in zip(rowNanIdx, colNanIdx):
            Ytmp[i, j] = np.dot(OmegaHat[i, :], xHat[:, j])

    # Estimate velocitiy
    vHat = np.zeros((nLF, Ytmp.shape[1]))
    vHat[:, 1:] = xHat[:, 1:] - xHat[:, 0:-1]
    vHat[:, 0] = vHat[:, 1] # Assume velocity is the same at t=0 and t=1. This guess shouldn't matter much.

    return np.dot(OmegaHat, xHat), np.asarray(np.bmat([OmegaHat, np.zeros((N, nLF))])), \
            np.asarray(np.bmat([[xHat], [vHat]]))

def kfSetupAug(Y, C, XHat, nLF, tVar=10):
    """
    Sets up matrices, initial state and initial covariance estimates for the Kalman filter
    """
    N, T = Y.shape

    # Estimate state variance from fit tVar observations
    Q = np.diag(np.var(detrend(XHat[:, 0:tVar]), axis=1))

    # Estimate observation variance. Initial guess shouldn't matter too much...
    obsVar = np.nanmean(np.nanvar(Y - np.dot(C[:, 0:nLF], XHat[0:nLF, :]), axis=1, ddof=1))
    R = np.diag(N*[obsVar if obsVar != 0 else np.mean(Q[0:nLF, 0:nLF]) * 0.1])
    # If there are no observations for a sensor, doesn't matter what gets filled in for its variance...
    #meanObsVar = np.nanmean(R)
    #R[np.isnan(R)] = meanObsVar

    pi0 = XHat[:, 0].copy().T
    sigma0 = Q

    return pi0, sigma0, Q, R

def ssmEMAug(Y, nLF, maxIt=50, dt=0.25):
    """
    Runs state space EM algorithm

    Arguments
    -Y: N x T numpy array
        Data array
    -maxIt: int
        Number of iterations to run of EM

    Returns
    -XHat, sigma_smooth, A, C, Q, R, pi0, sigma0
    """
    N, T = Y.shape

    # A is fixed
    A = np.asarray(np.bmat([[np.identity(nLF), dt*np.identity(nLF)], \
                            [np.zeros((nLF, nLF)), np.identity(nLF)]]))

    # Estimate SSM parameters with PCA, using EM to handle missing values
    _, C, pcaXHat = pcaEstMDAug(Y, nLF, maxIt=50)
    pi0, sigma0, Q, R = kfSetupAug(Y, C, pcaXHat, nLF)

    # Keep track of hidden state mean and covariance
    XHat = None
    sigma_smooth = None

    for i in range(maxIt):
        ##### E step
        # Estimate hidden state
        XHat, sigma_smooth, sigmaLag_smooth, YImp = kalmanSmooth(Y, pi0, sigma0, A, C, Q, R, 2*nLF)

        # Second moment
        P = np.zeros((T, 2*nLF, 2*nLF))
        for t in range(T):
            P[t, :, :] = sigma_smooth[t, :, :] + np.outer(XHat[:, t], XHat[:, t].T)

        # Lagged second moment
        PLag = np.zeros((T-1, 2*nLF, 2*nLF))
        for t in range(T-1):
            PLag[t, :, :] = sigmaLag_smooth[t, :, :] + np.outer(XHat[:, t+1], XHat[:, t].T)

        ##### M step
        Omega, Q, R, pi0, sigma0 = mStepAug(YImp, XHat, P, PLag, A)
        C = np.asarray(np.bmat([Omega, np.zeros((N, nLF))]))

    # Finally, re-estimate hidden state    
    XHat, sigma_smooth, _, _ = kalmanSmooth(Y, pi0, sigma0, A, C, Q, R, 2*nLF)

    return XHat, sigma_smooth, A, C, Q, R, pi0, sigma0


### Log likelihood
def logL(X, Y, pi0, sigma0, A, C, Q, R):
    logLs = np.zeros(Y.shape[1])

    for t in range(Y.shape[1]):
        # Observation contribution to likelihood
        obsRes = np.nan_to_num(Y[:, t] - np.dot(C, X[:, t]))

        logLs[t] = -0.5*np.dot(obsRes, np.dot(np.linalg.inv(R), obsRes))

        if t >= 1:
            # Hidden state contribution to likelihood
            dynamRes = X[:, t] - np.dot(A, X[:, t-1])
            logLs[t] = logLs[t] - 0.5*np.dot(dynamRes, np.dot(np.linalg.inv(Q), dynamRes))

    # Prior's contribution to likelihood
    logLs[0] = logLs[0] - 0.5*np.dot(X[:, 0] - pi0, np.dot(np.linalg.inv(sigma0), X[:, 0] - pi0))

    return logLs


