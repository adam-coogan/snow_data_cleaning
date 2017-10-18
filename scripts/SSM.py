"""Contains a class used to represent state space models.

Todo:
    * Implement switching
    * _pca_est assumes the data is already standardized/aligned, but I never
        standardize it...

"""

import numpy as np
from sklearn import linear_model


class SSM(object):
    """Represents a state space model.

    The model has a variety of methods available to estimate its parameters and
    the hidden state's values.

    Note:
        Regime switching is not fully supported yet. While the Kalman filter
        and smoother support it, the m-step methods and _ssm_setup_pca do not.

    Examples:
        Estimate the model parameters and hidden state for some observations:
        >>> ssm = SSM(y, u, v, numpy.array(T * [0]), [0], n_LF)
        >>> ssm.em(num_it)
        >>> ssm.get_y_hat()

        Run the Kalman smoother and get observation estimates using
        user-defined model parameters:
        >>> ssm = SSM(y, u, v, numpy.array(T * [0]), [0], n_LF)
        >>> ssm.As = [A]
        >>> ssm.Bs = [B]
        >>> ssm.Cs = [C]
        >>> ssm.Ds = [D]
        >>> ssm.Qs = [Q]
        >>> ssm.Rs = [R]
        >>> ssm.pi1s = [pi1]
        >>> ssm.sigma1s = [sigma1]
        >>> ssm.smooth()
        >>> ssm.get_y_hat()

    """

    def __init__(self, y, u, v, ss, s_list, n_LF):
        """Initialize state space model with observations.

        Args:
            y (N x T numpy.array): observation matrix
            u (L x T numpy.array): state transition controls
            v (M x T numpy.array): observation controls
            ss (N_S x T numpy.array): switching state values
            s_list (list of int): length N_S list of possible switching state
                values
            n_LF (int): number of latent factors to use to model the data

        """
        # Inputs
        #: N x T numpy.array: observation matrix
        self.y = y
        #: L x T numpy.array: state transition controls
        self.u = u
        #: M x T numpy.array: observation controls
        self.v = v
        #: N_S x T numpy.array: switching state values
        self.ss = ss
        #: list of int: length N_S list of possible switching state values
        self.s_list = s_list

        # Dimensions
        self.N, self.T = y.shape
        #: int: dimensionality of state transition control
        self.L = u.shape[0]
        #: int: dimensionality of observation transition control
        self.M = v.shape[0]
        #: int: number of latent factors to use to model the data
        self.n_LF = n_LF

        self._init_params()

    def _init_params(self):
        """Set Q to be the n_LF x n_LF identity matrix and the other model
        parameters to None."""
        #: N_S x n_LF x n_LF numpy.array: state transition matrix
        self.As = None
        #: N_S x n_LF x L numpy.array: state transition control matrix
        self.Bs = None
        #: N_S x N x n_LF numpy.array: state transition matrix
        self.Cs = None
        #: N_S x N x M numpy.array: state transition matrix
        self.Ds = None
        #: N_S x n_LF x n_LF numpy.array: state transition matrix. Fixed to be
        # the identity for numerical stability.
        self.Qs = np.array(len(self.s_list) * [np.identity(self.n_LF)])
        #: N_S x n_LF numpy.array: initial state prior mean
        self.pi1 = None
        #: N_S x n_LF x n_LF numpy.array: initial state prior covariance
        self.sigma1 = None

    def _kf_predict(self, t):
        """Run Kalman filter prediction step.

        Updates predicted mean x_pred and covariance sigma_pred.

        Args:
            t (int): time at which to run prediction step. Must be between 1
                and T-1.

        """
        # sigma_t|t-1
        self.sigma_pred[t-1] = np.dot(np.dot(self.As[self.ss[t]],
                                             self.sigma_filt[t-1]),
                                      self.As[self.ss[t]].T) \
                + self.Qs[self.ss[t]]

        # This is a bit of a hack. The better way to do this would be to use the
        # eigendecomposition of sigma.
        self.sigma_pred[t-1] = 0.5 * (self.sigma_pred[t-1]
                                      + self.sigma_pred[t-1].T)

        # x_t|t-1
        if self.u is None:
            self.x_pred[:, t-1] = np.dot(self.As[self.ss[t]],
                                         self.x_filt[:, t-1])
        else:
            self.x_pred[:, t-1] = np.dot(self.As[self.ss[t]],
                                         self.x_filt[:, t-1])\
                    + np.dot(self.Bs[self.ss[t]], self.u[:, t])

    def _kf_update(self, t):
        """Run Kalman filter update step

        Handles partial observations by zeroing out corresponding components of
        the observation vector and parameter matrices. If no observations were
        made, the predictive mean and covariance are used as the filtered ones.
        See pdf documentation of the method as well as Murphy 18.3.1.2 and
        Shumway and Stoffer.  x_filt and sigma_filt are updated.

        Args:
            t (int): time at which to run update step. Must be between 1 and
                T-1.

        """
        # Check whether there were any observations: get indices of sensors
        # with no observations
        nan_ss = np.where(np.isnan(self.y[:, t]))[0]

        # If there are some observations, go through with update step
        if nan_ss.size < self.N:
            # Zero out unobserved components (see Shumway and Stofer)
            y_obs = np.nan_to_num(self.y[:, t])
            # Zero out rows of parameter matrices corresponding to unobserved
            # components
            C_t = self.Cs[self.ss[t]].copy()
            C_t[nan_ss, :] = 0.0
            D_t = self.Ds[self.ss[t]].copy()
            D_t[nan_ss, :] = 0.0
            R_t = self.Rs[self.ss[t]].copy()
            R_t[nan_ss, :] = 0.0
            R_t[:, nan_ss] = 0.0
            R_t[nan_ss, nan_ss] = 1.0

            # Posterior predictive mean
            if self.v is None:
                y_pred = np.dot(C_t, self.x_pred[:, t-1])
            else:
                y_pred = np.dot(C_t, self.x_pred[:, t-1]) \
                        + np.dot(D_t, self.v[:, t])

            # Residual
            r = np.asarray(y_obs - y_pred)

            # Kalman Gain
            S = np.dot(np.dot(C_t, self.sigma_pred[t-1]), C_t.T) + R_t
            S_inverse = np.linalg.pinv(S)
            K = np.dot(np.dot(self.sigma_pred[t-1], C_t.T), S_inverse)

            # Correct the state mean
            self.x_filt[:, t] = self.x_pred[:, t-1] + np.dot(K, r)

            # Compute sigma_t|t using an EXPLICITY SYMMETRIC expression
            I_KC = np.identity(len(self.x_pred[:, t-1])) \
                    - np.dot(K, C_t)
            self.sigma_filt[t] = np.dot(I_KC, self.sigma_pred[t-1])
            # This is a bit of a hack. The better way to do this would be to use
            # the eigendecomposition of sigma.
            self.sigma_filt[t] = 0.5 * (self.sigma_filt[t] \
                                         + self.sigma_filt[t].T)
        else:
            # If there were no observations, don't adjust predictive mean or
            # covariance
            self.x_filt[:, t] = self.x_pred[:, t-1]
            self.sigma_filt[:, t] = self.sigma_pred[t-1]

    def filter(self):
        """ Runs the Kalman filter, updating the predictive and filtered means
        and covariances."""
        # Filtered mean and covariance x_t|t, sigma_t|t
        #: n_LF x T numpy.array: filtered means x_t|t. Indexed by t.
        self.x_filt = np.zeros([self.n_LF, self.T])
        self.x_filt[:, 0] = self.pi1.copy()
        #: T x n_LF x n_LF numpy.array: filtered covariances Sigma_t|t. Indexed
        # by t.
        self.sigma_filt = np.zeros([self.T, self.n_LF, self.n_LF])
        self.sigma_filt[0] = self.sigma1.copy()

        #: n_LF x T-1 numpy.array: predictive means x_t|t-1. Indexed by t-1.
        self.x_pred = np.zeros([self.n_LF, self.T-1])
        #: T-1 x n_LF x n_LF numpy.array: predictive covariances Sigma_t|t-1.
        # Indexed by t-1.
        self.sigma_pred = np.zeros([self.T-1, self.n_LF, self.n_LF])

        for t in range(1, self.T):
            self._kf_predict(t)
            self._kf_update(t)

    def smooth(self):
        """Run the Kalman smoother after running the Kalman filter.

        In addition to computing the smoothed means and covariances, the lagged
        covariance, smoothed distribution's second moment and lagged second
        moment are computed.

        """
        # Run filter first
        self.filter()

        # Smoothed mean and covariance, x_t|T and sigma_t|T
        #: n_LF x T numpy.array: smoothed means x_t|T. Indexed by t.
        self.x_smooth = np.zeros([self.n_LF, self.T])
        #: T x n_LF x n_LF numpy.array: smoothed covariances Sigma_t|T. Indexed
        # by t.
        self.sigma_smooth = np.zeros((self.T, self.n_LF, self.n_LF))

        #: T-1 x n_LF x n_LF numpy.array: smoothed lagged covariances
        # Sigma_{t,t-1}. Indexed by t-1!
        self.sigma_lag_smooth = np.zeros((self.T-1, self.n_LF, self.n_LF))

        # Initialize: sigma_T|T = sigma_T|T(filtered)
        self.sigma_smooth[-1] = self.sigma_filt[-1]
        # x_T|T = x_T|T(filtered)
        self.x_smooth[:, -1] = self.x_filt[:, -1]

        # sigmaLag_{T,T-1} = (1 - K_T C) A V_{T-1|T-1}, where K_T is Kalman
        # gain at last timestep.
        K_T = np.dot(self.sigma_pred[-1],
                     np.dot(self.Cs[self.ss[-1]].T,
                            np.linalg.pinv(np.dot(self.Cs[self.ss[-1]],
                                                  np.dot(self.sigma_pred[-1],
                                                         self.Cs[self.ss[-1]].T))
                                           + self.Rs[self.ss[-1]])))
        self.sigma_lag_smooth[-1] = np.dot(np.dot((np.identity(self.n_LF)
                                                   - np.dot(K_T, self.Cs[self.ss[-1]])),
                                                  self.As[self.ss[-1]]),
                                           self.sigma_filt[-2])

        # Backwards Kalman gain
        J = np.zeros((self.T-1, self.n_LF, self.n_LF))

        # Smoothing step. Runs from t=T-1 to t=0.
        for t in range(self.T-2, -1, -1):
            # Backward Kalman gain matrix
            J[t] = np.dot(np.dot(self.sigma_filt[t], self.As[self.ss[t]].T),
                          np.linalg.pinv(self.sigma_pred[t]))

            # Smoothed mean
            self.x_smooth[:, t] = self.x_filt[:, t] \
                    + np.dot(J[t], self.x_smooth[:, t+1] - self.x_pred[:, t])

            # Smoothed covariance. This is explicity symmetric.
            self.sigma_smooth[t] = self.sigma_filt[t] \
                    + np.dot(np.dot(J[t],
                                    self.sigma_smooth[t+1] - self.sigma_pred[t]),
                             J[t].T)

        # Lagged smoothed covariance (NOT symmetric!)
        for t in range(self.T-3, -1, -1):
            self.sigma_lag_smooth[t] = np.dot(self.sigma_filt[t+1], J[t].T) \
                    + np.dot(np.dot(J[t+1],
                                    self.sigma_lag_smooth[t+1]
                                    - np.dot(self.As[self.ss[t]],
                                             self.sigma_filt[t+1])),
                             J[t].T)

        #: T x n_LF x n_LF numpy.array: smoothed second moment P_t|T. Indexed by
        # t.
        self.P = self.sigma_smooth \
                + np.einsum("it,jt->tij",
                            self.x_smooth,
                            self.x_smooth)
        #: T-1 x n_LF x n_LF numpy.array: smoothed lagged second moment
        # P_{t,t-1}. Indexed by t-1!
        self.P_lag = self.sigma_lag_smooth \
                + np.einsum("it,jt->tij",
                            self.x_smooth[:, 1:],
                            self.x_smooth[:, :-1])

    def _pca_est(self, y_pca):
        """Estimate C and x using the PCA algorithm from Bai and Ng 2002.

        See Section 3 of Determining the Number of Factors in Approximate
        Factor Models, Jushan Bai and Serena Ng, Econometrica vol. 70 no. 1,
        Jan 2002 191-221 for details.

        Args:
            y_pca (N x T numpy.array): observation array with missing
                observations filled in.

        Returns:
            N x n_LF numpy.array, n_LF x T numpy.array: estimate of C and x.

        """
        # Observation covariance
        sigma_y = np.dot(y_pca, y_pca.T)
        # Get eigenvalues and vectors
        e_vals, e_vecs = np.linalg.eig(sigma_y)

        # Sort eigenvalues
        e_val_idx = np.argsort(e_vals)[::-1]

        # Get matrix of eigenvectors corresponding to n_LF largest eigenvalues
        C_pca = np.sqrt(self.N) * e_vecs[:, e_val_idx[0:self.n_LF]]

        # Use PCA result to estimate factors
        x_pca = np.dot(C_pca.T, y_pca) / float(self.N)

        return C_pca, x_pca

    def pca_est_MD(self, num_it):
        """Estimate C, x and missing observations using the PCA EM algorithm
        from Bai and Ng 2002.

        See the brief discussion in Section 7 of Determining the Number of
        Factors in Approximate Factor Models, Jushan Bai and Serena Ng,
        Econometrica vol. 70 no. 1, Jan 2002 191-221 for details.

        Args:
            num_it (int): number of EM iterations to run.

        Returns:
            N x n_LF numpy.array, n_LF x T numpy.array, N x T numpy.array:
                estimates of C, x and y with missing values filled in.

        """
        y_pca = self.y.copy()

        # Replace nans with average to start
        nan_ss, nan_ts = np.where(np.isnan(self.y))
        y_pca[nan_ss, nan_ts] = np.nanmean(self.y)

        C_pca, x_pca = None, None

        for i in range(0, num_it):
            # Get current lambda and F_t estimates
            C_pca, x_pca = self._pca_est(y_pca)

            # Replace nans with new estimates
            for i, t in zip(nan_ss, nan_ts):
                y_pca[i, t] = np.dot(C_pca[i, :], x_pca[:, t])

        return C_pca, x_pca, y_pca

    def _ssm_setup_pca(self, pca_num_it):
        """Initialize SSM parameters using PCA EM estimates.

        A is computed by linearly regressing each value of the hidden state
        onto the previous one. B, D, R, sigma1 are set to identity matrices of
        the correction dimension. pi1 is taken to be the first value of the
        hidden state estimated by PCA EM.

        Args:
            pca_num_it (int): numbed of PCA EM iterations to run to estimate C,
                hidden state and missing observations.

        """
        # Run PCA to estimate C, hidden state and missing observations
        # TODO: standardize data!
        C_pca, x_pca, y_pca = self.pca_est_MD(pca_num_it)
        self.Cs = np.array(len(self.s_list) * [C_pca])

        # Estimate A by regressing state onto its lag
        clf = linear_model.LinearRegression()
        clf.fit(x_pca[:, 0:-1].T, x_pca[:, 1:].T)
        self.As = np.array(len(self.s_list) * [clf.coef_])

        # Extract pi_1 from hidden state estimate
        self.pi1 = x_pca[:, 0]

        # Q is fixed to be the identity. Initial guesses for B, D, R and Sigma_1
        # shouldn't matter much
        self.Bs = np.array(len(self.s_list) * [np.ones([self.n_LF, self.L])])
        self.Ds = np.array(len(self.s_list) * [np.ones([self.N, self.M])])
        self.Rs = np.array(len(self.s_list) * [np.identity(self.N)])
        self.sigma1 = np.identity(self.n_LF)

    def _e_step(self):
        """Compute expectation values need to run EM using current parameters.

        Returns:
            N x T numpy.array, T x N x n_LF numpy.array, N x T numpy.array:
                E[y_{it}], E[y_{it} x_{jt}] and E[(y_{it})^2].

        """
        # Run filter and smoother
        self.smooth()

        # Get sensor and time indices for missing data
        nan_ss, nan_ts = np.where(np.isnan(self.y))
        # Fill in missing Y values
        E_y = self.y.copy()
        E_y[nan_ss, nan_ts] = (np.einsum("tij,jt->it",
                                         self.Cs[self.ss],
                                         self.x_smooth)
                               + np.einsum("tij,jt->it",
                                           self.Ds[self.ss],
                                           self.v))[nan_ss, nan_ts]

        E_y_x = np.einsum("it,jt->tij",
                          self.y,
                          self.x_smooth)
        E_y_x_unobs = np.einsum("tij,tjk->tik",
                                self.Cs[self.ss],
                                self.P) \
                + np.einsum("tij,jt,kt->tik",
                            self.Ds[self.ss],
                            self.v,
                            self.x_smooth)
        E_y_x[nan_ts, nan_ss, :] = E_y_x_unobs[nan_ts, nan_ss, :]

        E_y_y_diag = np.einsum("it,it->it", self.y, self.y)
        E_y_y_diag_unobs = np.square(np.einsum("tij,jt->it",
                                               self.Cs[self.ss],
                                               self.x_smooth)
                                     + np.einsum("tij,jt->it",
                                                 self.Ds[self.ss],
                                                 self.v)) \
                + np.einsum("tij,tjk,tik->it",
                            self.Cs[self.ss],
                            self.sigma_smooth,
                            self.Cs[self.ss]) \
                + np.einsum("tii->it", self.Rs[self.ss])
        E_y_y_diag[nan_ss, nan_ts] = E_y_y_diag_unobs[nan_ss, nan_ts]

        return E_y, E_y_x, E_y_y_diag

    def _m_step(self, E_y, E_y_x, E_y_y_diag):
        """Runs the m step, updating the model's parameter to their new values.

        Args:
            E_y (N x T numpy.array): E[y_{it}].
            E_y_x (T x N x n_LF numpy.array): E[y_{it} x_{jt}].
            E_y_y_diag (N x T numpy.array): E[(y_{it})^2].

        """
        # Loop over switching state values
        for s_idx, s in enumerate(self.s_list):
            self._m_step_A_B(s_idx, s)
            self._m_step_C_D(s_idx, s, E_y, E_y_x)
            self._m_step_R(s_idx, s, E_y, E_y_x, E_y_y_diag)
            self._m_step_pi1_sigma1()

    def _m_step_A_B(self, s_idx, s):
        """Runs the m step for A and B, updating them to their new values.

        Note:
            Currently only tested with s_idx and s set to 0.

        Args:
            s_idx (int): index of parameter lists corresponding to switching
                state.
            s (int): value of switching state during the regime of interest.

        """
        # Find T_s2 = {t | t >= 2 and s_t = s}
        T_s2 = np.where(self.ss == s)[0]
        if T_s2[0] == 0: # enforce t > 1!
            T_s2 = T_s2[1:]

        # Simultaneously solve for state transition matrix and control
        # transition matrix
        # [sum_{t=2}^T u_t u_t^T]^-1
        inv_sum_uu = np.linalg.pinv(np.dot(self.u[:, T_s2], self.u.T[T_s2, :]))
        # sum_{t=2}^T P_{t-1} = sum_{t=1}^{T-1} P_t
        sum_P_T_1 = np.sum(self.P[T_s2 - 1, :, :], axis=0)
        inv_sum_P_T_1 = np.linalg.pinv(sum_P_T_1)
        # sum_{t=2}^T x_t u_t^T
        sum_x_u = np.dot(self.x_smooth[:, T_s2], self.u.T[T_s2, :])
        # sum_{t=2}^T x_{t-1} u_t^T
        sum_x_1_u = np.dot(self.x_smooth[:, T_s2 - 1], self.u.T[T_s2, :])
        # sum_{t=2}^T P_{t,t-1}
        sum_P_lag = np.sum(self.P_lag[T_s2 - 1, :, :], axis=0)

        # Construct system to solve
        M_AB = np.asarray(np.bmat([[np.identity(self.n_LF),
                                    np.dot(sum_x_1_u, inv_sum_uu)],
                                   [np.dot(sum_x_1_u.T, inv_sum_P_T_1),
                                    np.identity(self.L)]]))
        N_AB = np.asarray(np.bmat([np.dot(sum_P_lag, inv_sum_P_T_1),
                                   np.dot(sum_x_u, inv_sum_uu)]))
        x_AB = np.linalg.solve(M_AB.T, N_AB.T).T
        # Extract A and B
        self.As[s_idx, :, :] = x_AB[:, 0:self.n_LF]
        self.Bs[s_idx, :, :] = x_AB[:, self.n_LF:]

    def _m_step_C_D(self, s_idx, s, E_y, E_y_x):
        """Runs the m step for C and D, updating them to their new values.

        Note:
            Currently only tested with s_idx and s set to 0.

        Args:
            s_idx (int): index of parameter lists corresponding to switching
                state.
            s (int): value of switching state during the regime of interest.
            E_y (N x T numpy.array): E[y_{it}].
            E_y_x (T x N x n_LF numpy.array): E[y_{it} x_{jt}].
            E_y_y_diag (N x T numpy.array): E[(y_{it})^2].

        """
        # Find T_s = {t | s_t = s}
        T_s = np.where(self.ss == s)[0]
        sum_x_v = np.dot(self.x_smooth[:, T_s], self.v[:, T_s].T)
        inv_sum_vv = np.linalg.pinv(np.dot(self.v[:, T_s], self.v[:, T_s].T))
        inv_sum_P = np.linalg.pinv(np.sum(self.P[T_s, :, :], axis=0))

        M_CD = np.asarray(np.bmat([[np.identity(self.n_LF),
                                    np.dot(sum_x_v, inv_sum_vv)],
                                   [np.dot(sum_x_v.T, inv_sum_P),
                                    np.identity(self.M)]]))
        N_CD = np.asarray(np.bmat([np.dot(np.sum(E_y_x[T_s, :, :], axis=0),
                                          inv_sum_P),
                                   np.dot(np.dot(E_y[:, T_s], self.v[:, T_s].T),
                                          inv_sum_vv)]))
        x_CD = np.linalg.solve(M_CD.T, N_CD.T).T

        # Extract C and D
        self.Cs[s_idx, :, :] = x_CD[:, 0:self.n_LF]
        self.Ds[s_idx, :, :] = x_CD[:, self.n_LF:]

    def _m_step_R(self, s_idx, s, E_y, E_y_x, E_y_y_diag):
        """Runs m step for R, updating it to its new value.

        Note:
            Must run have computed new values for C and D before calling this
            function. Currently only tested with s_idx and s set to 0.

        Args:
            s_idx (int): index of parameter lists corresponding to switching
                state.
            s (int): value of switching state during the regime of interest.
            E_y (N x T numpy.array): E[y_{it}].
            E_y_x (T x N x n_LF numpy.array): E[y_{it} x_{jt}].
            E_y_y_diag (N x T numpy.array): E[(y_{it})^2].

        """
        # Find {t | s_t = s}
        s_t = np.where(self.ss == s)[0]

        self.Rs[s_idx, :, :] = np.diag(np.sum(E_y_y_diag[:, s_t], axis=1)
                + np.sum(np.einsum("ij,tjk,ik->it",
                                   self.Cs[s_idx, :, :],
                                   self.P[s_t, :, :],
                                   self.Cs[s_idx, :, :]),
                         axis=1)
                + np.sum(np.square(np.dot(self.Ds[s_idx, :, :],
                                          self.v[:, s_t])),
                         axis=1)
                - 2.0 * np.sum(np.einsum("tij,ij->it",
                                         E_y_x[s_t, :, :],
                                         self.Cs[s_idx, :, :]),
                               axis=1)
                - 2.0 * np.einsum("it,it->i",
                                  np.dot(self.Ds[s_idx, :, :],
                                         self.v[:, s_t]),
                                  E_y[:, s_t])
                + 2.0 * np.einsum("it,it->i",
                                  np.dot(self.Cs[s_idx, :, :],
                                         self.x_smooth[:, s_t]),
                                  np.dot(self.Ds[s_idx, :, :],
                                         self.v[:, s_t]))) \
                / float(self.T)

    def _m_step_pi1_sigma1(self):
        """Runs the m step for pi1 and sigma1, updating them to their new
        values."""
        # Initial state mean
        self.pi1 = self.x_smooth[:, 0]

        # Initial state covariance
        self.sigma1 = self.P[0, :, :] - np.outer(self.x_smooth[:, 0],
                                                 self.x_smooth[:, 0].T)

    def em(self, num_it, pca_num_it=50):
        """Runs EM algorithm described in the algorithm notes file.

        The model's parameters are initialized using PCA EM. Each EM iteration
        consists of an e step that updates the filtering and smoothing
        quantities followed by the m step, which updates the model parameters.
        After the EM iterations, the smoother is run a final time using the last
        m-step parameters. See the notes in ssm_em.pdf for full details on the
        algorithm.

        Args:
            num_it (int): number of EM iterations to run. Must be non-negative.
            pca_num_it (int): number of PCA EM iterations to run to initialize
                the model's parameters. Should have little impact. Defaults to
                50.

        """
        # First estimate parameters using PCA
        self._ssm_setup_pca(pca_num_it)

        # Run EM algorithm
        for i in range(num_it):
            E_y, E_y_x, E_y_y_diag = self._e_step()
            self._m_step(E_y, E_y_x, E_y_y_diag)

        # Run smoother one last time
        self.smooth()

    def get_y_hat(self):
        """Use the current model parameters and smoothed means to compute
        expected observation for each sensor at each time."""
        return np.einsum("tij,jt->it", self.Cs[self.ss], self.x_smooth) \
                + np.einsum("tij,jt->it", self.Ds[self.ss], self.v)

