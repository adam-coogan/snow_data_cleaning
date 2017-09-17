import numpy as np
from Utilities import obsFrac, scale, descale
from kalman import Kalman
from scipy.stats import linregress
from scipy.linalg import block_diag
from scipy.signal import detrend
from sklearn import linear_model

##### Kalman smoother

def kalman_smooth(Y, U, V, ss, pi0, sigma0, As, Bs, Cs, Ds, Qs, Rs):
    """
    Runs Kalman filtering and smoothing step of dynamic factor model estimator.

    Arguments
    -Y: N x T
        Observations
    -U: L x T
        State controls
    -V: P x T
        Observation controls
    -pi0: n_LF x 1
    -sigma0: n_LF x n_LF
    -A: n_LF x n_LF
    -B: n_LF x L
    -C: N x n_LF
    -D: N x M
    -Q: n_LF x n_LF
    -R: N x N

    Returns
    -n_LF x T numpy array
        Smoothed means
    -T x n_LF x n_LF numpy array
        Smoothed covariances
    -T-1 x n_LF x n_LF numpy array
        Smoothed lagged covariances (ie, cov[x_t, x_t-1])
    """
    N, T = Y.shape
    n_LF = As[0].shape[0]

    # Initialize Kalman filter
    kf = Kalman(mu_0=pi0.copy(), sigma_0=sigma0.copy(), A=As[ss[0]], B=Bs[ss[0]], C=Cs[ss[0]], D=Ds[ss[0]],
            Q=Qs[ss[0]], R=Rs[ss[0]])

    # sigma_t|t, mu_t|t
    sigma_filt = np.zeros([T, n_LF, n_LF])
    sigma_filt[0] = sigma0.copy()
    mu_filt = np.zeros([T, n_LF])
    mu_filt[0] = pi0.copy()

    # sigma_t|t-1, mu_t|t-1. NOTE: indexed by t-1!!!
    sigma_pred = np.zeros([T-1, n_LF, n_LF])
    mu_pred = np.zeros([T-1, n_LF])

    # Filtering step
    for t in range(1, T):
        kf.A = As[ss[t]].copy()
        kf.B = Bs[ss[t]].copy()
        kf.C = Cs[ss[t]].copy()
        kf.D = Ds[ss[t]].copy()
        kf.Q = Qs[ss[t]].copy()
        kf.R = Rs[ss[t]].copy()

        ### Prediction step
        kf.predict(U[:, t])

        # Save mu_t|t-1 and sigma_t|t-1
        sigma_pred[t-1, :, :] = kf.sigma
        mu_pred[t-1] = kf.mu

        # Need to change observation, observation control and observation noise matrices if observation is
        # incomplete! See Shumway and Stoffer page 314.
        # First, figure out which sensors don't have observations:
        nan_sensors = np.where(np.isnan(Y[:, t]))[0]
        # If some observations were present, go through with the update step
        if nan_sensors.size < N:
            # Zero those observations
            y_t = np.nan_to_num(Y[:, t])

            # Modify parameters
            kf.C[nan_sensors, :] = 0.0
            kf.D[nan_sensors, :] = 0.0
            kf.R[nan_sensors, :] = 0.0
            kf.R[:, nan_sensors] = 0.0
            kf.R[nan_sensors, nan_sensors] = 1.0

            kf.update(y_t, V[:, t])

        # Save filtered mean, covariance
        sigma_filt[t, :, :] = kf.sigma
        mu_filt[t] = kf.mu

    # sigma_t|T, mu_t|T
    sigma_smooth = np.zeros((T, n_LF, n_LF))
    mu_smooth = np.zeros((T, n_LF))

    # Initialize: sigma_T|T = sigma_T|T(filtered)
    sigma_smooth[-1] = sigma_filt[-1]
    # mu_T|T = mu_T|T(filtered)
    mu_smooth[-1] = mu_filt[-1]

    # Lagged covariance. Indexed by t-1.
    sigma_lag_smooth = np.zeros((T-1, n_LF, n_LF))
    # sigmaLag_{T,T-1} = (1 - K_T C) A V_{T-1|T-1}, where K_T is Kalman gain at last timestep.
    # TODO: unclear what to do here if last observation contains missing components
    kf.A = As[ss[-1]].copy()
    kf.C = Cs[ss[-1]].copy()
    kf.R = Rs[ss[-1]].copy()
    K_T = np.dot(sigma_pred[-1], np.dot(kf.C.T, np.linalg.pinv(np.dot(kf.C, \
                                                                    np.dot(sigma_pred[-1], kf.C.T)) + kf.R)))
    sigma_lag_smooth[-1] = np.dot(np.dot((np.identity(n_LF) - np.dot(K_T, kf.C)), kf.A), sigma_filt[-2])

    # Backwards Kalman gain
    J = np.zeros((T-1, n_LF, n_LF))

    # Smoothing step. Runs from t=T-1 to t=0.
    for t in range(T-2, -1, -1):
        kf.A = As[ss[t]].copy()

        # Backward Kalman gain matrix
        J[t] = np.dot(np.dot(sigma_filt[t], kf.A.T), np.linalg.pinv(sigma_pred[t]))

        # Smoothed mean
        mu_smooth[t] = mu_filt[t] + np.dot(J[t], mu_smooth[t+1] - mu_pred[t])

        # Smoothed covariance. This is explicity symmetric.
        sigma_smooth[t, :, :] = sigma_filt[t] + np.dot(np.dot(J[t], sigma_smooth[t+1] - sigma_pred[t]), J[t].T)

    # Lagged smoothed covariance (NOT SYMMETRIC!)
    for t in range(T-3, -1, -1):
        kf.A = As[ss[t]].copy()

        sigma_lag_smooth[t, :, :] = np.dot(sigma_filt[t+1], J[t].T) \
                    + np.dot(np.dot(J[t+1], (sigma_lag_smooth[t+1] - np.dot(kf.A, sigma_filt[t+1]))), J[t].T)

    # Fill in missing Y values
    Y_imp = Y.copy()
    nan_sensors, nan_times = np.where(np.isnan(Y_imp))
    Y_imp[nan_sensors, nan_times] = (np.einsum("tij,tj->it", Cs[ss], mu_smooth) \
            + np.einsum("tij,jt->it", Ds[ss], V))[nan_sensors, nan_times]

    return mu_smooth.T, sigma_smooth, sigma_lag_smooth, Y_imp, sigma_filt

##### M step of SSM EM algorithm
# Takes Q = I, R diagonal. TODO: set eigenvalues of A <= 1!
def m_step_stable(Y, Y_imp, U, V, ss, s_list, X_hat, sigma_smooth, P, P_lag, pi0_olds, sigma0_olds, A_olds,
        B_olds, C_olds, D_olds, Q_olds, R_olds):
    """
    Estimates SSM parameters given state and covariance estimates.

    Arguments
    -Y_imp: N x T
        Data matrix with nan values filled in by Kalman smoother
    -U: L x T
        State control matrix
    -V: M x T
        Observation control matrix
    -X_hat: n_LF x T
        Hidden state means E[x_t | y]
    -P: T x n_LF x n_LF
        Covariance estimate: E[x_t x_t^T | y]
    -P_lag: T-1 x n_LF x n_LF
        Lagged covariance estimate: E[x_t x_{t-1}^T | y]

    Returns
    -Estimates for A, B, C, D and R
    """
    N, T = Y_imp.shape
    n_LF = X_hat.shape[0]
    L = U.shape[0]
    M = V.shape[0]

    nan_sensors, nan_times = np.where(np.isnan(Y))

    # Need these expectation values to handle missing observations
    # TODO: move to e-step method
    E_y = Y_imp.copy()

    E_y_x = np.einsum("it,jt->tij", Y, X_hat)
    E_y_x_unobs = np.einsum("tij,tjk->tik", C_olds[ss], P) \
            + np.einsum("tij,jt,kt->tik", D_olds[ss], V, X_hat)
    E_y_x[nan_times, nan_sensors, :] = E_y_x_unobs[nan_times, nan_sensors, :]

    E_y_y_diag = np.einsum("it,it->it", Y, Y)
    E_y_y_diag_unobs = np.square(np.einsum("tij,jt->it", C_olds[ss], X_hat) 
            + np.einsum("tij,jt->it", D_olds[ss], V)) \
            + np.einsum("tij,tjk,tik->it", C_olds[ss], sigma_smooth, C_olds[ss]) \
            + np.einsum("tii->it", R_olds[ss])
    E_y_y_diag[nan_sensors, nan_times] = E_y_y_diag_unobs[nan_sensors, nan_times]

    # Make new parameters for each possible value of the switching state
    A_news = np.zeros([len(s_list)] + list(A_olds[0].shape))
    B_news = np.zeros([len(s_list)] + list(B_olds[0].shape))
    C_news = np.zeros([len(s_list)] + list(C_olds[0].shape))
    D_news = np.zeros([len(s_list)] + list(D_olds[0].shape))
    R_news = np.zeros([len(s_list)] + list(R_olds[0].shape))

    for s_idx, s in enumerate(s_list):
        ##### Find A and B
        # Find {t | t >= 2 and s_t = s}
        s_t2 = np.where(ss == s)[0]
        if s_t2[0] == 0: # enforce t > 1!
            s_t2 = s_t2[1:]

        # Simultaneously solve for state transition matrix and control transition matrix
        inv_sum_uu = np.linalg.pinv(np.dot(U[:, s_t2], U.T[s_t2, :])) # [sum_{t=2}^T u_t u_t^T]^-1
        sum_P_T_1 = np.sum(P[s_t2 - 1, :, :], axis=0) # sum_{t=2}^T P_{t-1} = sum_{t=1}^{T-1} P_t
        inv_sum_P_T_1 = np.linalg.pinv(sum_P_T_1)
        sum_x_u = np.dot(X_hat[:, s_t2], U.T[s_t2, :]) # sum_{t=2}^T x_t u_t^T
        sum_x_1_u = np.dot(X_hat[:, s_t2 - 1], U.T[s_t2, :]) # sum_{t=2}^T x_{t-1} u_t^T
        sum_P_lag = np.sum(P_lag[s_t2 - 1, :, :], axis=0) # sum_{t=2}^T P_{t,t-1}

        # Construct system to solve
        A_AB = np.asarray(np.bmat([[np.identity(n_LF), np.dot(sum_x_1_u, inv_sum_uu)],
                                    [np.dot(sum_x_1_u.T, inv_sum_P_T_1), np.identity(L)]]))
        b_AB = np.asarray(np.bmat([np.dot(sum_P_lag, inv_sum_P_T_1), np.dot(sum_x_u, inv_sum_uu)]))
        x_AB = np.linalg.solve(A_AB.T, b_AB.T).T
        # Extract A and B
        A_news[s_idx, :, :] = x_AB[:, 0:n_LF]
        B_news[s_idx, :, :] = x_AB[:, n_LF:]

        ##### Find C and D
        # Find {t | s_t = s}
        s_t = np.where(ss == s)[0]
        sum_x_v = np.dot(X_hat[:, s_t], V[:, s_t].T)
        inv_sum_vv = np.linalg.pinv(np.dot(V[:, s_t], V[:, s_t].T))
        inv_sum_P = np.linalg.pinv(np.sum(P[s_t, :, :], axis=0))

        A_CD = np.asarray(np.bmat([[np.identity(n_LF), np.dot(sum_x_v, inv_sum_vv)],
                                    [np.dot(sum_x_v.T, inv_sum_P), np.identity(M)]]))
        b_CD = np.asarray(np.bmat([np.dot(np.sum(E_y_x[s_t, :, :], axis=0), inv_sum_P),
            np.dot(np.dot(E_y[:, s_t], V[:, s_t].T), inv_sum_vv)]))
        x_CD = np.linalg.solve(A_CD.T, b_CD.T).T

        # Extract A and B
        C_news[s_idx, :, :] = x_CD[:, 0:n_LF]
        D_news[s_idx, :, :] = x_CD[:, n_LF:]

        ##### Compute R
        R_news[s_idx, :, :] = np.diag((np.sum(E_y_y_diag[:, s_t], axis=1)
                + np.sum(np.einsum("ij,tjk,ik->it", C_news[s_idx, :, :], P[s_t, :, :], C_news[s_idx, :, :]),
                    axis=1)
                + np.sum(np.square(np.dot(D_news[s_idx, :, :], V[:, s_t])), axis=1)
                - 2.0 * np.sum(np.einsum("tij,ij->it", E_y_x[s_t, :, :], C_news[s_idx, :, :]), axis=1)
                - 2.0 * np.einsum("it,it->i", np.dot(D_news[s_idx, :, :], V[:, s_t]), E_y[:, s_t])
                + 2.0 * np.einsum("it,it->i", np.dot(C_news[s_idx, :, :], X_hat[:, s_t]),
                    np.dot(D_news[s_idx, :, :], V[:, s_t]))) / float(T))

    # Initial state mean
    pi0_new = X_hat[:, 0]

    # Initial state covariance
    sigma0_new = P[0, :, :] - np.outer(X_hat[:, 0], X_hat[:, 0].T)

    return pi0_new, sigma0_new, A_news, B_news, C_news, D_news, np.array([np.identity(n_LF)] * len(s_list)), \
            R_news

##### Initialization functions
def pca_est(Y, n_LF):
    """
    Performs PCA fit to C, X assuming no missing data using the method from from Bai+Ng 2002. Assumes the data
    is already standardized/preprocessed.

    Arguments
    -Y: N x T numpy array
        Data array
    -n_LF: int
        Number of latent variables to use

    Returns
    -N x T numpy array
        Estimate of Y from estimated factors and loadings
    -N x n_LF numpy array
        Estimate of observation matrix
    -n_LF x T numpy array
        Estimate of hidden state
    """
    N = Y.shape[0]

    # Observation covariance
    sigmaY = np.dot(Y, Y.T)
    # Find eigenvalues/vectors
    eVals, eVecs = np.linalg.eig(sigmaY)
    # Sort eigenvalues
    eValIdx = np.argsort(eVals)[::-1]

    # Get matrix of eigenvectors corresponding to n_LF largest eigenvalues
    C = np.sqrt(N) * eVecs[:, eValIdx[0:n_LF]]

    # Use PCA result to estimate factors
    X = np.dot(C.T, Y) / float(N)

    return C, X

def pca_est_MD(Y, n_LF, max_it=10):
    """
    Performs PCA fit to C, X using the missing data strategy from Bai+Ng 2002. Assumes the data is already
    standardized/preprocessed.

    Arguments
    -Y: N x T numpy array
        Data array
    -n_LF: int
        Number of latent variables to use

    Returns
    -N x n_LF numpy array
        Estimate of C
    -n_LF x T numpy array
        Estimate of X
    -N x T numpy array
        Y with missing values imputed with PCA
    """
    Y_tmp = Y.copy()
    N = Y_tmp.shape[0]

    # Replace nans with average to start
    rowNanIdx, colNanIdx = np.where(np.isnan(Y))
    Y_tmp[(rowNanIdx, colNanIdx)] = np.nanmean(Y)

    C, X = None, None

    for i in range(0, max_it):
        # Get current lambda and F_t estimates
        C, X = pca_est(Y_tmp, n_LF)

        # Replace nans with new estimates
        for i, j in zip(rowNanIdx, colNanIdx):
            Y_tmp[i, j] = np.dot(C[i, :], X[:, j])

    return C, X, Y_tmp

def ssm_setup(Y, U, V, C, X, n_LF):
    """
    Sets up matrices, initial state and initial covariance estimates for the Kalman filter

    Arguments:
    -Y: N x T float array
        Data matrix
    -C: N x n_LF float array
        Loadings matrix
    -X: n_LF x T float array
        Hidden state estimate
    -n_LF: int
        Number of latent factors

    Returns
    -n_LF x n_LF
        Hidden state transition matrix (A)
    -n_LF x L
        B
    -n_LF x M
        D
    -n_LF x n_LF numpy array
        Hidden state noise estimate (Q)
    -N x N numpy array
        Observation noise estimate (R)
    -n_LF numpy array
        Initial hidden state estimate (pi0)
    -n_LF x n_LF numpy array
        Initial hidden state covariance estimate (sigma0)
    """
    N, T = Y.shape

    # Estimate A by regressing state onto its lag
    clf = linear_model.LinearRegression()
    clf.fit(X[:, 0:-1].T, X[:, 1:].T)
    A = clf.coef_

    Q = np.identity(n_LF)

    # Estimate observation variance. Initial guess shouldn't matter too much...
    #obs_var = np.nanmean(np.nanvar(Y - np.dot(C, X), axis=1, ddof=1))
    R = np.identity(N) #np.diag(N*[obs_var if obs_var != 0 else 1.0])

    pi0 = X[:, 0].copy().T
    sigma0 = Q

    return A, np.ones([n_LF, U.shape[0]]), np.ones([N, V.shape[0]]), Q, R, pi0, sigma0

def ssm_em_stable(Y, U, V, ss, s_list, n_LF, max_it, pca_max_it=50):
    """
    Runs state space EM algorithm

    Arguments
    -Y: N x T numpy array
        Data array
    -max_it: int
        Number of iterations to run of EM

    Returns
    -X_hat, sigma_smooth, A, C, Q, R, pi0, sigma0
    """
    N, T = Y.shape

    # Estimate SSM parameters with PCA, using EM to handle missing values. First, shift the time series:
    C, X_PCA, Y_imp = pca_est_MD(Y, n_LF, pca_max_it)

    # Use PCA results to set up the SSM
    A, B, D, Q, R, pi0, sigma0 = ssm_setup(Y_imp, U, V, C, X_PCA, n_LF)

    As = np.array([A] * len(s_list))
    Bs = np.array([B] * len(s_list))
    Cs = np.array([C] * len(s_list))
    Ds = np.array([D] * len(s_list))
    Qs = np.array([Q] * len(s_list))
    Rs = np.array([R] * len(s_list))

    # Keep track of hidden state mean and covariance
    X_hat = None
    sigma_smooth = None

    for i in range(max_it):
        ##### E step
        # Estimate hidden state
        X_hat, sigma_smooth, sigma_lag_smooth, Y_imp, sigma_filt = kalman_smooth(Y, U, V, ss, pi0, sigma0, As,
                Bs, Cs, Ds, Qs, Rs)

        # Second moment
        P = np.zeros((T, n_LF, n_LF))
        for t in range(T):
            P[t, :, :] = sigma_smooth[t, :, :] + np.outer(X_hat[:, t], X_hat[:, t].T)

        # Lagged second moment
        P_lag = np.zeros((T-1, n_LF, n_LF))
        for t in range(T-1):
            P_lag[t, :, :] = sigma_lag_smooth[t, :, :] + np.outer(X_hat[:, t+1], X_hat[:, t].T)

        ##### M step
        pi0, sigma0, As, Bs, Cs, Ds, Qs, Rs = m_step_stable(Y, Y_imp, U, V, ss, s_list, X_hat, sigma_smooth,
                P, P_lag, pi0, sigma0, As, Bs, Cs, Ds, Qs, Rs)

    # Finally, re-estimate hidden state
    # TODO: should I use Y_imp here?
    X_hat, sigma_smooth, _, _, sigma_filt = kalman_smooth(Y, U, V, ss, pi0, sigma0, As, Bs, Cs, Ds, Qs, Rs)

    return X_hat, sigma_smooth, sigma_filt, pi0, sigma0, As, Bs, Cs, Ds, Qs, Rs


