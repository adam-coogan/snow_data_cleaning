"""Contains a class used to represent state space models where the hidden state
is augmented by its derivative.

Todo:
    * _pca_est assumes the data is already standardized/aligned, but I never
        standardize it...

"""

import numpy as np


class Augmented_SSM(object):
    """Represents a state space model.

    The model has a variety of methods available to estimate its parameters and
    the hidden state's values.

    Note:
        Regime switching is not fully supported yet. While the Kalman filter
        and smoother support it, the m-step methods and _ssm_setup_pca do not.

    Examples:
        Estimate the model parameters and hidden state for some observations:
        >>> ssm = SSM(y, dt, n_HS, u, v, numpy.array(T * [0]))
        >>> ssm.em(num_it)
        >>> ssm.get_y_hat()

        Run the Kalman smoother and get observation estimates using
        user-defined model parameters:
        >>> ssm = SSM(y, dt, n_HS, u, v, numpy.array(T * [0]))
        >>> ssm.Bs = numpy.array([B])
        >>> ssm.Cs = numpy.array([C])
        >>> ssm.Ds = numpy.array([D])
        >>> ssm.Qs = numpy.array([Q])
        >>> ssm.Rs = numpy.array([R])
        >>> ssm.pi1s = numpy.array([pi1])
        >>> ssm.sigma1s = numpy.array([sigma1])
        >>> ssm.smooth()
        >>> ssm.get_y_hat()

    Attributes:
        y (N x T numpy.array): observation matrix
        u (L x T numpy.array): state transition controls
        v (M x T numpy.array): observation controls
        ss (1 x T numpy.array): switching state values. The values of the
            switching state must be integers between 0 and the number of
            switching states N_s minus one, without skipping values.
        N_s (int): number of switching states (ie, number of unique values in
            ss).
        L (int): dimensionality of state transition control
        M (int): dimensionality of observation transition control
        n_HS (int): number of hidden states to use to model the data. Note that
            the number of latent factors will be 2*n_HS since the states are
            augmented by their derivatives.
        A (2n_HS x 2n_HS numpy.array): state transition matrix. This is just
            the finite difference matrix.
        Bs (N_s x 2n_HS x L numpy.array): state transition control matrices
        Cs (N_s x N x 2n_HS numpy.array): observation matrices
        Ds (N_s x N x M numpy.array): observation control matrices
        Qs (N_s x N x N numpy array): observation noise covariance matrices.
            Set to be the identity for numerical stability.
        Rs (N_s x 2n_HS x 2n_HS numpy array): state noise covariance matrices.
            Taken to be diagonal for numerical stability.
        pi1 (2n_HS numpy.array): initial state prior mean
        sigma1 (2n_HS x 2n_HS numpy.array): initial state prior covariance
    """

    def __init__(self, y, dt, n_HS, u=None, v=None, ss=None):
        """Initialize state space model with observations.

        Args:
            y (N x T numpy.array): observation matrix
            dt (float): timestep size.
            n_HS (int): number of hidden states to use to model the data
            u (L x T numpy.array): state transition controls. Defaults to zeros.
            v (M x T numpy.array): observation controls. Defaults to ones.
            ss (N_s x T numpy.array): switching state values. Defaults to
                zeros.

        """
        self.y = y
        self.N, self.T = y.shape
        self.dt = dt
        self.n_HS = n_HS

        # If u isn't provided, don't bias the state transitions
        if u is None:
            self.u = np.zeros([1, self.T])
        else:
            self.u = u
        self.L = self.u.shape[0]

        # If v isn't provide, let each sensor have its own constant bias
        if v is None:
            self.v = np.ones([1, self.T])
        else:
            self.v = v
        self.M = self.v.shape[0]

        if ss is None:
            self.ss = np.zeros(self.T, dtype=int)
        else:
            self.ss = ss

        # Number of switching states
        self.N_s = np.unique(ss).size

        # Fix A to be the finite difference matrix
        self.A = np.asarray(np.bmat([[np.identity(self.n_HS),
                                      self.dt * np.identity(self.n_HS)],
                                     [np.zeros(2 * [self.n_HS]),
                                      np.identity(self.n_HS)]]))

    def _kf_predict(self, t):
        """Run Kalman filter prediction step.

        Updates predicted mean x_pred and covariance sigma_pred.

        Args:
            t (int): time at which to run prediction step. Must be between 1
                and T-1.

        """
        # sigma_t|t-1
        self.sigma_pred[t-1] = np.dot(np.dot(self.A, self.sigma_filt[t-1]),
                                      self.A.T) \
                + self.Qs[self.ss[t]]

        # This is a bit of a hack. The better way to do this would be to use the
        # eigendecomposition of sigma.
        self.sigma_pred[t-1] = 0.5 * (self.sigma_pred[t-1]
                                      + self.sigma_pred[t-1].T)

        # x_t|t-1
        if self.u is None:
            self.x_pred[:, t-1] = np.dot(self.A, self.x_filt[:, t-1])
        else:
            self.x_pred[:, t-1] = np.dot(self.A, self.x_filt[:, t-1])\
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
        #: 2n_HS x T numpy.array: filtered means x_t|t. Indexed by t.
        self.x_filt = np.zeros([2*self.n_HS, self.T])
        self.x_filt[:, 0] = self.pi1.copy()
        #: T x 2n_HS x 2n_HS numpy.array: filtered covariances Sigma_t|t.
        # Indexed by t.
        self.sigma_filt = np.zeros([self.T, 2*self.n_HS, 2*self.n_HS])
        self.sigma_filt[0] = self.sigma1.copy()

        #: 2n_HS x T-1 numpy.array: predictive means x_t|t-1. Indexed by t-1.
        self.x_pred = np.zeros([2*self.n_HS, self.T-1])
        #: T-1 x 2n_HS x 2n_HS numpy.array: predictive covariances Sigma_t|t-1.
        # Indexed by t-1.
        self.sigma_pred = np.zeros([self.T-1, 2*self.n_HS, 2*self.n_HS])

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
        #: 2n_HS x T numpy.array: smoothed means x_t|T. Indexed by t.
        self.x_smooth = np.zeros([2*self.n_HS, self.T])
        #: T x 2n_HS x 2n_HS numpy.array: smoothed covariances Sigma_t|T. Indexed
        # by t.
        self.sigma_smooth = np.zeros((self.T, 2*self.n_HS, 2*self.n_HS))

        #: T-1 x 2n_HS x 2n_HS numpy.array: smoothed lagged covariances
        # Sigma_{t,t-1}. Indexed by t-1!
        self.sigma_lag_smooth = np.zeros((self.T-1, 2*self.n_HS, 2*self.n_HS))

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
        self.sigma_lag_smooth[-1] = np.dot(np.dot((np.identity(2*self.n_HS)
                                                   - np.dot(K_T, self.Cs[self.ss[-1]])),
                                                  self.A),
                                           self.sigma_filt[-2])

        # Backwards Kalman gain
        J = np.zeros((self.T-1, 2*self.n_HS, 2*self.n_HS))

        # Smoothing step. Runs from t=T-1 to t=0.
        for t in range(self.T-2, -1, -1):
            # Backward Kalman gain matrix
            J[t] = np.dot(np.dot(self.sigma_filt[t], self.A.T),
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
                                    - np.dot(self.A, self.sigma_filt[t+1])),
                             J[t].T)

        #: T x 2n_HS x 2n_HS numpy.array: smoothed second moment P_t|T. Indexed
        # by t.
        self.P = self.sigma_smooth \
                + np.einsum("it,jt->tij",
                            self.x_smooth,
                            self.x_smooth)
        #: T-1 x 2n_HS x 2n_HS numpy.array: smoothed lagged second moment
        # P_{t,t-1}. Indexed by t-1!
        self.P_lag = self.sigma_lag_smooth \
                + np.einsum("it,jt->tij",
                            self.x_smooth[:, 1:],
                            self.x_smooth[:, :-1])

    def _pca_est(self, y_pca):
        """Estimate Omega and d using the PCA algorithm from Bai and Ng 2002.

        See Section 3 of Determining the Number of Factors in Approximate
        Factor Models, Jushan Bai and Serena Ng, Econometrica vol. 70 no. 1,
        Jan 2002 191-221 for details.

        Args:
            y_pca (N x T numpy.array): observation array with missing
                observations filled in.

        Returns:
            N x n_HS numpy.array, n_HS x T numpy.array: estimate of C and d.

        """
        # Observation covariance
        sigma_y = np.dot(y_pca, y_pca.T)
        # Get eigenvalues and vectors
        e_vals, e_vecs = np.linalg.eig(sigma_y)

        # Sort eigenvalues
        e_val_idx = np.argsort(e_vals)[::-1]

        # Get matrix of eigenvectors corresponding to n_HS largest eigenvalues
        Omega_pca = np.sqrt(self.N) * e_vecs[:, e_val_idx[0:self.n_HS]]

        # Use PCA result to estimate factors
        d_pca = np.dot(Omega_pca.T, y_pca) / float(self.N)

        return Omega_pca, d_pca

    def pca_est_MD(self, pca_num_it):
        """Estimate Omega, d and missing observations using the PCA EM
        algorithm from Bai and Ng 2002.

        See the brief discussion in Section 7 of Determining the Number of
        Factors in Approximate Factor Models, Jushan Bai and Serena Ng,
        Econometrica vol. 70 no. 1, Jan 2002 191-221 for details.

        Args:
            pca_num_it (int): number of EM iterations to run.

        Returns:
            N x n_HS numpy.array, n_HS x T numpy.array, N x T numpy.array:
                estimates of Omega, d and y with missing values filled in.

        """
        y_pca = self.y.copy()

        # Replace nans with average to start
        nan_ss, nan_ts = np.where(np.isnan(self.y))
        y_pca[nan_ss, nan_ts] = np.nanmean(self.y)

        for i in range(0, pca_num_it):
            # Get current lambda and F_t estimates
            Omega_pca, d_pca = self._pca_est(y_pca)

            # Replace nans with new estimates
            for i, t in zip(nan_ss, nan_ts):
                y_pca[i, t] = np.dot(Omega_pca[i, :], d_pca[:, t])

        return Omega_pca, d_pca, y_pca

    def _ssm_setup_pca(self, pca_num_it):
        """Initialize SSM parameters using PCA EM estimates.

        B, D, R, sigma1 are set to identity matrices of the correction
        dimension. pi1 is taken to be the first value of the hidden state
        estimated by PCA EM. C is taken to be (Omega_pca 0_{N x n_HS}), where
        Omega_pca is the loading matrix estimated using PCA EM.

        Notes:
            Need to standardize the data!

        Args:
            pca_num_it (int): number of PCA EM iterations to run to estimate C,
                hidden state and missing observations.

        """
        # TODO: standardize data!
        # Run PCA to estimate C, hidden state and missing observations
        Omega_pca, d_pca, y_pca = self.pca_est_MD(pca_num_it)
        C = np.asarray(np.bmat([Omega_pca,
                                np.zeros([self.N, self.n_HS])]))
        self.Cs = np.array(self.N_s * [C])

        # Extract pi_1 from hidden state estimate using second data point to
        # estimate the state's derivative
        self.pi1 = np.concatenate([d_pca[:, 0], d_pca[:, 1] - d_pca[:, 0]])

        # Initial guesses for B, D, Q, R and Sigma_1 shouldn't matter much
        self.Bs = np.array(self.N_s * [np.ones([2 * self.n_HS, self.L])])
        self.Ds = np.array(self.N_s * [np.ones([self.N, self.M])])
        self.Qs = np.array(self.N_s * [np.identity(2 * self.n_HS)])
        self.Rs = np.array(self.N_s * [np.identity(self.N)])
        self.sigma1 = np.identity(2 * self.n_HS)

    def _e_step(self):
        """Compute expectation values need to run EM using current parameters.

        Returns:
            N x T numpy.array, T x N x 2n_HS numpy.array, N x T numpy.array:
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
            E_y_x (T x N x 2n_HS numpy.array): E[y_{it} x_{jt}].
            E_y_y_diag (N x T numpy.array): E[(y_{it})^2].

        """
        # Loop over switching state values
        for s in range(self.N_s):
            self._m_step_B(s)
            self._m_step_C_D(s, E_y, E_y_x)
            self._m_step_Q(s)
            self._m_step_R(s, E_y, E_y_x, E_y_y_diag)
            self._m_step_pi1_sigma1()

    def _m_step_B(self, s):
        """Runs the m step for B, updating it to its new value.

        Args:
            s (int): value of switching state during the regime of interest.

        """
        # Find T_s2 = {t | t >= 2 and s_t = s}
        T_s2 = np.where(self.ss == s)[0]
        if T_s2[0] == 0: # enforce t > 1!
            T_s2 = T_s2[1:]

        # [sum_{t=2}^T u_t u_t^T]^-1
        inv_sum_uu = np.linalg.pinv(np.dot(self.u[:, T_s2], self.u.T[T_s2, :]))

        # sum_{t=2}^T E[x_t] u_t^T
        sum_x_u = np.dot(self.x_smooth[:, T_s2], self.u.T[T_s2, :])

        # sum_{t=2}^T A E[x_{t-1}] u_t^T
        sum_A_x_1_u = np.dot(np.dot(self.A,
                                  self.x_smooth[:, T_s2 - 1]),
                           self.u.T[T_s2, :])

        self.Bs[s, :, :] = np.dot(sum_x_u - sum_A_x_1_u, inv_sum_uu)

    def _m_step_C_D(self, s, E_y, E_y_x):
        """Runs the m step for C and D, updating them to their new values. C
        will have the form (Omega 0_{N x n_HS}), where Omega is an N x n_HS
        matrix.

        Note:
            Currently only tested with s set to 0.

        Args:
            s (int): value of switching state during the regime of interest.
            E_y (N x T numpy.array): E[y_{it}].
            E_y_x (T x N x 2n_HS numpy.array): E[y_{it} x_{jt}].

        """
        # Find T_s = {t | s_t = s}
        T_s = np.where(self.ss == s)[0]


        # [sum_{t=1}^T E[d_t d_t^T]]^-1, where d_{1:n_HS,t} = x_{1:n_HS,t} (ie,
        # d is the vector of the state's value, excluding its derivatives)
        inv_sum_dd = np.linalg.pinv(np.sum(self.P[T_s, :self.n_HS, :self.n_HS],
                                           axis=0))

        # [sum_{t=1}^T v_t v_t^T]^-1
        inv_sum_vv = np.linalg.pinv(np.dot(self.v[:, T_s], self.v[:, T_s].T))

        # sum_{t=1}^T E[y_t d_t^T]: E_y_x[T_s, :, :self.n_HS]
        sum_d_v = np.dot(self.x_smooth[:self.n_HS, T_s], self.v[:, T_s].T)

        # Create the system of equations
        M_OmegaD = np.asarray(np.bmat([[np.identity(self.n_HS),
                                        np.dot(sum_d_v, inv_sum_vv)],
                                       [np.dot(sum_d_v.T, inv_sum_dd),
                                        np.identity(self.M)]]))

        N_OmegaD = np.asarray(np.bmat([np.dot(np.sum(E_y_x[T_s, :, :self.n_HS],
                                                     axis=0),
                                              inv_sum_dd),
                                       np.dot(np.dot(E_y[:, T_s],
                                                     self.v[:, T_s].T),
                                             inv_sum_vv)]))

        # Solve the system of equations
        x_OmegaD = np.linalg.solve(M_OmegaD.T, N_OmegaD.T).T
        # Stack Omega to obtain C = (Omega, 0_{N x n_HS})
        self.Cs[s, :, :] = np.asarray(np.bmat([x_OmegaD[:, :self.n_HS],
                                               np.zeros([self.N, self.n_HS])]))
        self.Ds[s, :, :] = x_OmegaD[:, self.n_HS:]

    def _m_step_R(self, s, E_y, E_y_x, E_y_y_diag):
        """Runs m step for R, updating it to its new value.

        Note:
            Must run have computed new values for C and D before calling this
            function. Currently only tested with s set to 0.

        Args:
            s (int): value of switching state during the regime of interest.
            E_y (N x T numpy.array): E[y_{it}].
            E_y_x (T x N x 2n_HS numpy.array): E[y_{it} x_{jt}].
            E_y_y_diag (N x T numpy.array): E[(y_{it})^2].

        """
        # Find {t | s_t = s}
        T_s = np.where(self.ss == s)[0]

        self.Rs[s, :, :] = np.diag(np.sum(E_y_y_diag[:, T_s], axis=1)
                + np.sum(np.einsum("ij,tjk,ik->it",
                                   self.Cs[s, :, :],
                                   self.P[T_s, :, :],
                                   self.Cs[s, :, :]),
                         axis=1)
                + np.sum(np.square(np.dot(self.Ds[s, :, :], self.v[:, T_s])),
                         axis=1)
                - 2.0 * np.sum(np.einsum("tij,ij->it",
                                         E_y_x[T_s, :, :],
                                         self.Cs[s, :, :]),
                               axis=1)
                - 2.0 * np.einsum("it,it->i",
                                  np.dot(self.Ds[s, :, :],
                                         self.v[:, T_s]),
                                  E_y[:, T_s])
                + 2.0 * np.einsum("it,it->i",
                                  np.dot(self.Cs[s, :, :],
                                         self.x_smooth[:, T_s]),
                                  np.dot(self.Ds[s, :, :],
                                         self.v[:, T_s]))) \
                / float(T_s.size)

    def _m_step_Q(self, s):
        # Find T_s2 = {t | t >= 2 and s_t = s}
        T_s2 = np.where(self.ss == s)[0]
        if T_s2[0] == 0: # enforce t > 1!
            T_s2 = T_s2[1:]

        self.Qs[s, :, :] = np.sum(self.P[T_s2, :, :]
                                  - np.einsum("ij,tkj->tik",
                                              self.A,
                                              self.P_lag[T_s2 - 1, :, :])
                                  - np.einsum("tij,jk->tik",
                                              self.P_lag[T_s2 - 1, :, :],
                                              self.A.T)
                                  + np.einsum("ij,tjk,kl->til",
                                              self.A,
                                              self.P[T_s2 - 1, :, :],
                                              self.A.T)
                                  + np.einsum("it,jt->tij",
                                              np.dot(self.Bs[s, :, :],
                                                     self.u[:, T_s2]),
                                              np.dot(self.A,
                                                     self.x_smooth[:, T_s2 - 1])
                                              - self.x_smooth[:, T_s2])
                                  + np.einsum("it,jt->tij",
                                              np.dot(self.A,
                                                     self.x_smooth[:, T_s2 - 1])
                                              - self.x_smooth[:, T_s2],
                                              np.dot(self.Bs[s, :, :],
                                                     self.u[:, T_s2]))
                                  + np.einsum("it,jt->tij",
                                              np.dot(self.Bs[s, :, :],
                                                     self.u[:, T_s2]),
                                              np.dot(self.Bs[s, :, :],
                                                     self.u[:, T_s2])),
                                  axis=0) / float(T_s2.size)

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

    def print_params(self):
        """Print state space model's parameters."""
        for s in range(self.N_s):
            print "A = " + str(self.A)
            print "B_%i = " % s + str(self.Bs[s])
            print "C_%i = " % s + str(self.Cs[s])
            print "D_%i = " % s + str(self.Ds[s])
            print "Q_%i = " % s + str(self.Qs[s])
            print "R_%i = " % s + str(self.Rs[s])
            print "\n"

        print "pi_1 = " + str(self.pi1)
        print "sigma_1 = " + str(self.sigma1)

