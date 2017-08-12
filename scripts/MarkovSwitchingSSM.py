import numpy as np 
from np.random import beta as beta_sample
from np.random import multivariate_normal as mvn_sample
from np.linalg import pinv
from np.linalg import det
from scipy.stats import invwishart
from scipy.stats import multivariate_normal as mvn_pdf

"""
State transition probabilities:

    p = p(s_t=1 | s_{t-1}=1)
    q = p(s_t=0 | s_{t-1}=0).

Transition matrix is therefore

    p(s_t = j | s_{t-1} = i) = [[q, 1 - p], [1 - q, p]].

TODO: generalize to >2 discrete states
TODO: make it possible to fix model parameters to be independent of hidden state
"""

class MarkovSwitchingSSM(object):
    """
    Implements a fully Bayesian Markov switching state space model
    """

    def __init__(self, y_ts, n_states, p_p, p_q, p_AQ, p_CR):
        """
        :y_ts: T x N matrix of observations
        :n_states: number of states in the switching variable's domain
        :p_p: dict with parameters a and b for p's prior, Beta(a, b)
        :p_q: dict with parameters a and b for p's prior, Beta(a, b)
        :p_pi0_sigma0: dict with prior parameters for pi_0 and sigma_0, which are mu_0, lambda, Psi and nu.
            The prior is 
                pi_0, sigma_0 ~ NIW(mu_0, lambda, Psi, nu),
            with nu > n_LF + 1.
        :p_AQ: list of dicts with prior parameters for A and Q, which are V_0, nu_0, B_0 and lambda_0. The
            prior is
                Q ~ inv-Wishart(V_0, nu_0),
                vec(A^T) | Q ~ N(vec(B_0^T), S \oprod lambda_0).
        :p_CR: list of dicts with prior parameters for C and R, which are V_0, nu_0, B_0 and lambda_0. The
            prior is
                R ~ inv-Wishart(V_0, nu_0),
                vec(C^T) | R ~ N(vec(B_0^T), S \oprod lambda_0).
        """
        self.y_ts = y_ts
        self.n_states = n_states
        self.p_p = p_p
        self.p_q = p_q
        self.p_pi0_sigma0 = p_pi0_sigma0
        self.p_AQ = p_AQ
        self.p_CR = p_CR
        # Number of observations and sensors
        self.T, self.N = y_ts.shape
        # Number of hidden states
        self.n_LF = self.p_AQ["B_0"].shape[0]
        # TODO: run some sanity checks on variable dimensions, values, positive definiteness, etc

    def _gibbs_sampler_init(self):
        """
        Initializes model parameters and generates a sample path for the switching variable

        TODO: unsure whether I'm handling the switching variable initialization in a smart way
        """
        # Draw from parameter's priors
        p = beta_sample(self.p_p["a"], self.p_p["b"])
        q = beta_sample(self.p_q["a"], self.p_q["b"])
        sigma_0 = invwishart.rvs(self.p_pi0_sigma0["nu"], self.p_pi0_sigma0["Psi"])
        pi_0 = mvn_sample(self.p_pi0_sigma0["mu_0"], sigma_0 / self.p_pi0_sigma0["lambda"])
        Qs = [invwishart.rvs(prior["nu_0"], prior["V_0"]) for prior in self.p_AQ]
        As = [mvn_sample(prior["B_0"].T.ravel(),
                            np.kron(Q, np.linalg.pinv(prior["lambda_0"]))).reshape(prior["B_0"].T.shape).T
                            for Q, prior in zip(Qs, self.p_AQ)]
        Cs = [mvn_sample(prior["B_0"].T.ravel(),
                            np.kron(R, np.linalg.pinv(prior["lambda_0"]))).reshape(prior["B_0"].T.shape).T
                            for R, prior in zip(Rs, self.p_CR)]

        # Transition matrix p(s_t | s_{t-1}) (s_t indexes rows, s_{t-1} indexes columns)
        p_transition = np.asarray([[q, 1.0-p], [1.0-q, p]])

        # Take p(s_0) to be steady state ones
        p_s0 = np.asarray([(1.0 - p) / (2.0 - p - q), (1.0 - q) / (2.0 - p - q)])
        s_0 = 0 if np.random.uniform() < p_s0[0] else 1

        # Draw s_1 | s_0 separately
        s_ts = np.zeros(self.y_ts.shape[0])
        s_0 = 0 if np.random.uniform() < p_transition[0, s_0] else 1

        # Draw s_1, ..., s_T
        for t in range(1, self.T):
            s_ts[t] = 0 if np.random.uniform() < p_transition[0, s_ts[t-1]] else 1

        return s_0, s_ts, p, q, pi_0, sigma_0, As, Cs, Qs, Rs

    def gibbs_sample(self, n_iter):
        """
        Run Gibbs sampler

        :n_iter: number of iterations to run Gibbs sampler
        :returns: list whose elements are the draws of [x_0, x_ts, s_0, s_ts, As, Cs, Rs, p, q] for each
            iteration.
        """
        results = {"x_0": [], "x_t": [], "s_0": [], "s_ts": [], "p": [], "q": [], "pi_0": [], "sigma_0": [], \
                "As": [], "Cs": [], "Qs": [], "Rs": []}

        # Perform "0th" draw
        s_0, s_ts, p, q, pi_0, sigma_0, As, Cs, Qs, Rs = self._gibbs_sampler_init()

        for i in range(1, n_iter):
            # Sample hidden state
            x_0, x_ts = self.draw_xs(s_0, s_ts, pi_0, sigma_0, As, Cs, Qs, Rs)
            
            # Sample switching state
            s_0, s_ts = self.draw_ss(x_0, x_ts, As, Cs, Qs, Rs, p, q)

            # Sample model parameters
            p, q = self.draw_PQ(s_0, s_ts)
            pi_0, sigma_0 = self.draw_pi0_sigma0(x_0)
            As, Qs = self.draw_AQ(x_0, x_ts, s_ts)
            Cs, Rs = self.draw_CR(x_ts, s_ts)

            # Save results
            results["x_0"].append(x_0)
            results["x_ts"].append(x_ts)
            results["s_0"].append(s_0)
            results["s_ts"].append(s_ts)
            results["p"].append(p)
            results["q"].append(q)
            results["pi_0"].append(pi_0)
            results["sigma_0"].append(sigma_0)
            results["As"].append(As)
            results["Cs"].append(Cs)
            results["Qs"].append(Qs)
            results["Rs"].append(Rs)

        return results

    def _filter_xs(self, s_0, s_ts, pi_0, sigma_0, As, Cs, Qs, Rs):
        """
        Computes x_1|1, P_1|1, ..., x_T|T, P_T|T given observations, parameters and values of switching
        variable at each time.

        :returns:
        -n_LF x T numpy array
            Smoothed means
        -T x n_LF x n_LF numpy array
            Smoothed covariances
        -T-1 x n_LF x n_LF numpy array
            Smoothed lagged covariances (ie, cov[x_t, x_t-1])
        """
        # Initialize Kalman filter using values of parameters at t = 0, even though they're never used
        kf = Kalman(mu_0=pi_0.copy(), sigma_0=sigma_0.copy(), A=As[s_0], B=None, C=Cs[s_0], D=None,
                    Q=Qs[s_0], R=Rs[s_0])

        # x_t|t, P_t|t, t = 1, ..., T
        x_filts = np.zeros([self.T, self.n_LF])
        P_filts = np.zeros([self.T, self.n_LF, self.n_LF])

        # x_t|t-1, P_t|t-1. t = 1, ..., T. Indexed by t.
        x_pred = np.zeros([self.T, self.n_LF])
        P_pred = np.zeros([self.T, self.n_LF, self.n_LF])

        # Filtering step
        for t in range(0, self.T): # corresponds to t = 1, ..., T
            # Change parameters. Never need to use A_{s_0}, etc.
            kf.A = As[s_ts[t]]
            kf.C = Cs[s_ts[t]]
            kf.Q = Qs[s_ts[t]]
            kf.R = Rs[s_ts[t]]

            # Compute x_{t|t-1}, P_{t|t-1}
            kf.predict()
            x_pred[t] = kf.mu
            P_pred[t] = kf.sigma

            # Compute x_{t|t}, P_{t|t}
            kf.update(self.y_ts[t])
            x_filts[t] = kf.mu
            P_filts[t] = kf.sigma

        # TODO: run smoother to fill in missing data!!!

        return x_filts, P_filts, mu_pred, sigma_pred

    def draw_xs(self, s_0, s_ts, pi_0, sigma_0, As, Cs, Qs, Rs):
        """
        Uses multimove sampling to draw x_0, ..., x_T from p(x_{0:T} | y_{1:T}, s_{0:T})

        :returns: x_0 and x_{1:T}
        """
        # Run Kalman filter to get x_{t|t}, P_{t|t}, t = 1, ..., T
        x_filts, P_filts, _, _ = self._filter_xs(self.y_ts, s_0, s_ts, pi_0, sigma_0, As, Cs, Qs, Rs)

        # Draw hidden states
        x_ts = np.zeros(x_filts.shape) # T, n_LF

        # x_T ~ N(x_{T|T}, P_{T|T})
        x_ts[-1] = mvn_sample(x_filts[-1], P_filts[-1])

        # Sampling step. Runs from t=T-1 to t=1.
        for t in range(self.T-2, -1, -1):
            # P_{t|t} A_{s_{t+1}}^T(A_{s_{t+1}} P_{t|t} A_{s_{t+1}}^T + Q_{s_{t+1}})^-1
            B = np.dot(P_filts[t], np.dot(As[s_ts[t+1]], \
                    np.linalg.pinv(np.dot(As[s_ts[t+1]], np.dot(P_filts[t], As[s_ts[t+1]].T)) + Qs[s_ts[t+1]])))

            # x_{t|t, x_{t+1}} = x_{t|t} + B (x_{t+1} - A_{s_t} x_{t|t})
            x_lookahead = x_filts[t] + np.dot(B, x_ts[t+1] - np.dot(As[s_ts[t+1]], x_filts[t]))))
            # P_{t|t, x_{t+1}} = (I - B A_{s_{t+1}}) P_{t|t}
            P_lookahead = P_filts[t] - np.dot(B, np.dot(As[s_ts[t+1]], P_filts[t]))

            # x_t ~ N(x_{t|t, x_{t+1}}, P_{t|t, x_{t+1}})
            x_draws[t] = mvn_sample(x_lookahead, P_lookahead)

        # Use x_0|0 = pi_0, P_0|0 = sigma_0
        B = np.dot(sigma_0, np.dot(As[s_ts[0]], \
                np.linalg.pinv(np.dot(As[s_ts[0]], np.dot(sigma_0, As[s_ts[0]].T)) + Qs[s_ts[0]])))
        # x_{0|0, x_1} = pi_0 + B (x_1 - A_{s_1} pi_0)
        x_lookahead = pi_0 + np.dot(B, x_ts[0] - np.dot(As[s_ts[0]], pi_0))))
        # P_{0|0, x_1} = (I - B A_{s_1}) sigma_0
        P_lookahead = sigma_0 - np.dot(B, np.dot(As[s_ts[0]], sigma_0))
        # Draw x_0
        x_0 = mvn_sample(x_lookahead, P_lookahead)

        return x_0, x_ts

    def _filter_ss(self, x_0, x_ts, As, Cs, Qs, Rs, p, q):
        """
        Runs Hamilton's filtering algorithm to compute p(s_t | x_{1:t}, y_{1:t}). p(s_0) is taken to be the steady
        state probabilities for the states.
        """
        # Transition matrix p(s_t | s_{t-1}) (s_t indexes rows, s_{t-1} indexes columns)
        p_transition = np.asarray([[q, 1.0-p], [1.0-q, p]])

        # Initial probability p(s_0) given by steady state probabilities
        p_s0 = np.asarray([(1.0 - p) / (2.0 - p - q), (1.0 - q) / (2.0 - p - q)])

        # Store p(s_t | x_{1:t}, y_{1:t}), t = 1, ..., T
        p_st = np.zeros([self.T, 2])

        # Need to treat t = 1 case separately since it depends on the t = 0 distribution
        # Joint distribution p(x_1, y_1 | s_1, x_0)
        p_xt_yt = np.tile([mvn_pdf(x_ts[0], np.dot(As[st], x_0), Qs[st]) \
                            * mvn_pdf(self.y_ts[0], np.dot(Cs[st], x_ts[0]), Rs[st]) for st in range(1)], 
                            (2, 1)).T
        # Previous conditional p(s_0)
        p_st_prev = np.tile(p_s0, (2, 1))
        # Joint distribution p(s_1, s_0 | x_1, y_1)
        p_st_joint = np.multiply(p_xt_yt, np.multiply(p_st_prev, p_transition))
        # Normalize, then marginalize over s_0 by summing over columns
        p_st[0] = np.sum(p_st_joint / p_st_joint.sum(), axis=1)

        # Filter to get distribution at times t = 2, ..., T
        for t in range(1, self.T):
            # Joint distribution p(x_t, y_t | s_t, x_{t-1})
            p_xt_yt = np.tile([mvn_pdf(x_ts[t], np.dot(As[st], x_ts[t-1]), Qs[st]) \
                                * mvn_pdf(self.y_ts[:, t], np.dot(Cs[st], x_ts[t]), Rs[st])
                                for st in range(1)], (2, 1)).T
            
            # Previous conditional p(s_{t-1} | x_{1:t-1}, y_{1:t-1})
            p_st_prev = np.tile(p_st[t-1], (2, 1))

            # Joint distribution p(s_t, s_{t-1} | x_{1:t}, y_{1:t})
            p_st_joint = np.multiply(p_xt_yt, np.multiply(p_st_prev, p_transition))

            # Normalize, then marginalize over s_{t-1} by summing over columns
            p_st[t] = np.sum(p_st_joint / p_st_joint.sum(), axis=1)

        return p_s0, p_st

    def draw_ss(self, x_0, x_ts, As, Cs, Qs, Rs, p, q):
        """
        Uses multimove sampling to draw s_0, ..., s_T from p(s_{0:T} | x_{0:T}, y_{1:T})
        """
        # Get p(s_0),p(s_t | x_{1:t}, y_{1:t}) for t = 1, ..., T
        p_s0, p_st = self._filter_ss(self.y_ts, x_0, x_ts, As, Cs, Qs, Rs, p, q)

        # Store s_t, t = 1, ..., T
        s_ts = np.zeros(self.T)
        # Sample s_T ~ p(s_T | x_{1:T}, y_{1:T})
        s_ts[-1] = 0 if np.random.uniform() < p_st[-1, 0] else 1

        # Draw s_{T-1}, ..., s_1
        for t in range(self.T-2, -1, -1):
            # Unnormalized p(s_t | x_{1:t}, y_{1:t}, s_{t+1}) \propto p(s_t | x_{1:t}, y_{1:t}) p(s_{t+1}|s_t)
            p_st_lookahead = p_st[t] * np.asarray([q, 1.0-p] if s_ts[t+1] == 0 else [1.0-q, p])
            # Normalize to obtain the probability mass function
            p_st_lookahead = p_st_lookahead / np.sum(p_st_lookahead)
            
            # Draw s_t from this Bernoulli distribution
            s_ts[t] = 0 if np.random.uniform() < s_lookahead[0] else 1

        # Handle s_0 draw separately:
        # p(s_0 | s_1) \propto p(s_0) p(s_1 | s_0)
        p_st_lookahead = p_s0 * np.asarray([q, 1.0-p] if s_ts[0] == 0 else [1.0-q, p])
        # Normalize to obtain the probability mass function
        p_st_lookahead = p_st_lookahead / np.sum(p_st_lookahead)

        # Draw s_0 from this Bernoulli distribution
        s_0 = 0 if np.random.uniform() < s_lookahead[0] else 1

        return s_0, s_ts

    def draw_PQ(self, s_0, s_ts):
        """
        Draws from posteriors for p | s_{0:T} and q | s_{0:T}

        :returns: draw from the posteriors
                p | s_{0:T} ~ Beta(a + n_{11}, b + n_{10})
                q | s_{0:T} ~ Beta(a + n_{00}, b + n_{01}),
            where n_{ij} = |{t : s_{t-1} = i, s_t = j}| and a and b are different for p and q.
        """
        # Store transition counts. n_ij = |{t : s_{t-1} = i, s_t = j}|
        n = np.zeros(2*[len(s_0)])
        
        # Handle first case separately
        n[s_0, s_ts[0]] = n[s_0, s_ts[0]] + 1

        # Count other transitions
        for t in range(1, len(s_ts)):
            n[s_ts[t-1], s_ts[t]] = n[s_ts[t-1], s_ts[t]] + 1

        # Draw p | s_{0:T} ~ Beta(u_11 + n_11, u_10 + n_10)
        p = beta_sample(self.p_p["a"] + n[1, 1], self.p_p["b"] + n[1, 0])
        # Draw q | s_{0:T} ~ Beta(u_00 + n_00, u_01 + n_01)
        q = beta_sample(self.p_q["a"] + n[0, 0], self.p_q["b"] + n[0, 1])

        return p, q

    def draw_pi0_sigma0(self, x_0):
        """
        Draws from pi_0, sigma_0 | x_0

        :returns: draw from the posterior
                pi_0, sigma_0 | x_0 ~ NIW(mu_1, lambda_1, Psi_1, nu_1),
            where here the posterior's parameters are
                mu_1 = (lambda mu_0 + x_0) / (lambda + 1)
                lambda_1 = lambda + 1
                nu_1 = nu + 1
                Psi_1 = Psi + lambda / (lambda + 1) [x_0 - mu_0]^T [x_0 - mu_0].
        """
        # Compute posterior's parameters
        mu_1 = (self.p_pi0_sigma0["lambda"] * self.p_pi0_sigma0["mu_0"] + x_0) \
                                                                        / (self.p_pi0_sigma0["lambda"] + 1.0)
        lambda_1 = self.p_pi0_sigma0["lambda"] + 1.0
        nu_1 = self.p_pi0_sigma0["nu"] + 1.0
        Psi_1 = self.p_pi0_sigma0["Psi"] + self.p_pi0_sigma0["lambda"] / (self.p_pi0_sigma0["lambda"] + 1.0) \
                                                            * np.square(x_0 - self.p_pi0_sigma0["mu_0"]).sum()

        # Draw from the inv-Wishart
        sigma_0 = invwishart.rvs(nu_1, Psi_1)
        # Draw from the normal
        pi_0 = mvn_sample(mu_1, sigma_0 / lambda_1)

        return pi_0, sigma_0

    def draw_AQ(self, x_0, x_ts, s_ts):
        """
        Draws from p(A_i, Q_i | x_{0:T}, s_{1:T}), where i runs over the domain of s_t.
        """
        return self._draw_multivariate_regression(X=np.concatenate([x_0, x_ts[:-1]]), Y=x_ts, s_ts, self.p_AQ,
                                                    self.n_states)

    def draw_CR(self, x_ts, s_ts):
        """
        Draws from p(C_i, R_i | x_{1:T}, y_{1:T}, s_{1:T}), where i runs over the domain of s_t.
        """
        return self._draw_multivariate_regression(X=x_ts, Y=self.y_ts, s_ts, self.p_CR, self.n_states)

    def _draw_multivariate_regression(self, X, Y, s_ts, priors):
        """
        Consider a model

            Y_i = X_i B_i^T + E_i,

        where Y_i is a T_i x N matrix of observation from times at which the switch state s_t was equal to i,
        X_i is a T_i x k matrix of regressors, B_i is a k x N matrix of regression coefficients and E_i is a
        T_i x N matrix of errors with rows from N(0, S_i). This function draws from the posteriors
        p(B_i, S_i | X, Y).

        Arguments (with the subscript i on parameters dropped for clarity):
        -priors: a list of length n_states containing dicts with the prior's parameters, which are V_0, nu_0,
            B_0 and lambda_0. The prior is

                S ~ inv-Wishart(V_0, nu_0)
                vec(B^T) | S ~ N(vec(B_0^T), S \oprod lambda_0),

            where \oprod is the Kronecker product.
        
        :returns: draw of B, S | X, Y, s_{1:T} from

                S | X, Y, s_{1:T} ~ inv-Wishart(V_n, nu_n)
                vec(B^T) | X, Y, s_{1:T} ~ N(vec(B_n^T), S \oprod lambda_n^-1),

            where vec(A) is the vector constructed by concatenating A's columns and

                nu_n = nu_0 + T
                lambda_n = X^T X + lambda_0
                B_n^T = (X^T X + lambda_0)^{-1} (X^T Y + lambda_0 B_0^T)
                V_n = V_0 + (Y - X B_n^T)^T (Y - X B_n^T) + (B_n^T - B_0^T) lambda_0 (B_n^T - B_0^T).
        """
        # Need to draw from posterior for each value in the domain of s_t
        Bs = []
        Ss = []

        for i in range(self.n_states):
            # Times at which s_t = i
            ts = np.where(s_ts == i)[0]
            # {y_t | s_t = i}
            ys = Y[ts]
            # {x_t | s_t = i}
            xs = X[ts]

            # Compute posterior parameters
            nu_n = priors[i]["nu_0"] + len(ts)

            lambda_n = priors[i]["lambda_0"] + np.dot(xs.T, xs)

            B_n = np.dot(np.linalg.pinv(np.dot(xs.T, xs) + priors[i]["lambda_0"]), \
                            np.dot(xs.T, ys) + np.dot(priors[i]["lambda_0"], np.priors[i]["B_0"].T)).T

            V_n = priors[i]["V_0"] + np.dot((ys - np.dot(xs, B_n.T)).T, ys - np.dot(xs, B_n.T)) \
                    + np.dot((B_n.T - prior[i]["B_0"].T).T, \
                                np.dot(prior[i]["lambda_0"], B_n.T - prior[i]["B_0"].T))

            # Draw from posterior!
            Ss.append(invwishart.rvs(nu_n, V_n))
            Bs.append(mvn_sample(B_n.T.ravel(), \
                                    np.kron(Ss[-1], np.linalg.pinv(lambda_n))).reshape(B_n.T.shape).T)

        return Bs, Ss

