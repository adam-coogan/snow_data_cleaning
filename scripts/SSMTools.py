import numpy as np
from Utilities import obsFrac, scale, descale, alignSeries
from kalman import Kalman
from scipy.stats import linregress
from scipy.linalg import block_diag
from scipy.signal import detrend
from sklearn import linear_model

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

