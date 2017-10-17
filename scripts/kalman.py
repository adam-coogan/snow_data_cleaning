import numpy as np 
from numpy.linalg import pinv
from numpy.linalg import det

class Kalman():
    '''
    Implements a 1D (observation dimension) Bayesian Kalman filter following the probabilistic approach of Murphy page ~641.  
       dim_z is the number of measurement inputs
    
    Attributes
    ----------
    mu : numpy.array(dim_mu, 1)
        State estimate vector

    sigma : numpy.array(dim_x, dim_x)
        Covariance matrix
    
    A : numpy.array(dim_mu, dim_mu)
        State Transition matrix
    
    B : numpy.array(dim_mu, dim_u)
        Control transition matrix
    
    C : numpy.array(dim_mu, dim_mu)
        Measurement function
    
    D : numpy.array(dim_mu, dim_u)
        Control observation matrix
    
    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix
        
    Q : numpy.array(dim_x, dim_x)
        Process noise matrix
        
    S : numpy.array(dim_z, dim_z)
        Observation Noise Estimate. For now set to R 
    '''

    def __init__(self, mu_0, sigma_0, A, B, C, D, Q, R, state_labels=None):
        '''
        dim_mu = state dimension
        dim_y  = observation dimension
        dim_u  = control dimension

        Parameters
        ----------
        mu_0 : numpy.array(dim_mu, 1)
            Initial state estimate vector

        sigma_0 : numpy.array(dim_mu, dim_mu)
            Initial covariance matrix

        A : numpy.array(dim_mu, dim_mu)
            State Transition matrix

        B : numpy.array(dim_mu, dim_u)
            Control transition matrix

        C : numpy.array(dim_y, dim_mu)
            Measurement function

        D : numpy.array(dim_mu, dim_u)
            Control observation matrix

        R : numpy.array(dim_y, dim_y)
            Measurement noise matrix

        Q : numpy.array(dim_mu, dim_mu)
            Process noise matrix

        state_labels : list(dim_mu)
            Labels the state vector by name.  Unused other than conveinience. 
        '''
        self.A = A   # Parameter matrix A 
        self.B = B   # Parameter matrix B 
        self.C = C   # Parameter matrix C 
        self.D = D   # Parameter matrix D
        self.Q = Q   # State noise covaraiance matrix 
        self.R = R   # Observation noise covariance matrix
        self.S = self.R # Observation Noise Estimate. For now set to R 
        self.mu = mu_0 # Initial state estimate 
        self.sigma = sigma_0 # Initial state covariance 
        self.state_labels = state_labels
        self.K = None

    def predict(self, u=None): 
        ''' Predict step for the Kalman filter.  See Murphy Sec. 18.3.1.1
        
        Parameters
        ----------
        u : np.array(dim_u)
            control input at the current timestep

        '''
        
        # Here we need to check for missing values.  If an observation is missing,
        # then the transition matrix A needs to be modified to impute the missing values.
        # Differentials and differential velocites should not be imputed, but just maintain the track. 
        # Eventually, A might be some non-stationary matrix which depends on the present system state 
        # or external inputs such as temp/cloud-cover/etc.. 
        # if nan_mask is not None

        # sigma_{t|t-1}
        self.sigma = np.dot(np.dot(self.A, self.sigma), self.A.T) + self.Q
        # This is a bit of a hack. The better way to do this would be to use the eigendecomposition of sigma.
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

        # mu_{t|t-1}
        if u is None:
            self.mu = np.dot(self.A, self.mu)
        else:
            self.mu = np.dot(self.A, self.mu) + np.dot(self.B, u)

    def update(self, Y, v=None):
        '''
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.  Murphy Sec. 18.3.1.2
        
        Parameters
        ----------
        Y : np.array(dim_z)
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.
        V: observation control
        '''
        # Posterior predictive mean 
        if v is None:
            self.y = np.dot(self.C, self.mu)
        else:
            self.y = np.dot(self.C, self.mu) + np.dot(self.D, v)
        
        r = np.asarray(Y - self.y) # residual 
        # If the residual is an NaN (observation invalid), set it to zero so there is no update
        # I.e. r=0 -> K=0 for that sensor 
        # TODO: Revisit this since it will impact log-likelihood
        #nan_idx = np.where(np.isnan(r))[0]
        r[np.isnan(r)] = 0

        S = np.dot(np.dot(self.C, self.sigma), self.C.T) + self.R #         
        S_inverse = np.nan
        
        try:
            S_inverse = pinv(S)
        except: 
            return 'nan'

        # Kalman Gain 
        K = np.dot(np.dot(self.sigma, self.C.T), S_inverse)

        # Correct the state covariance and mean 
        self.mu = self.mu + np.dot(K, r)
        # Compute sigma_t|t using an EXPLICITY SYMMETRIC expression
        #self.sigma = np.dot(I_KC, np.dot(self.sigma, I_KC.T)) + np.dot(np.dot(K, self.R), K.T)
        I_KC = np.identity(len(self.mu)) - np.dot(K, self.C)
        self.sigma = np.dot(I_KC, self.sigma)
        # This is a bit of a hack. The better way to do this would be to use the eigendecomposition of sigma.
        self.sigma = 0.5 * (self.sigma + self.sigma.T)
        
        # Update the class attribute values 
        # TODO: this seems unnecessary
        self.K = K 
        self.S = S 

        # Gaussian log-likeliehood 
        #loglikelihood = -len(r)/2.*np.log(2*np.pi)-np.log(det(S))/2.-np.dot(np.dot(r, S_inverse), r.T)/2.
        #return loglikelihood

def EstimateObservationNoise(series, start_obs=2000, end_obs=3000):
    obs = series[start_obs:end_obs] 
    obs = obs[~np.isnan(obs)]
    obs = obs - np.mean(obs)
    var = np.std(obs)**2

    return var

def obsNoiseDetrend(series, start_obs=2000, end_obs=3000):
    obs = series[start_obs:end_obs].values.copy()
    # Indices of non-nans
    obsIdx = np.where(~np.isnan(obs))[0]
    # Observed values
    obs = obs[obsIdx]
    print obs.shape

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(obsIdx.reshape((obsIdx.shape[0],1)), obs.reshape((obs.shape[0],1)))
    # Compute trend
    trend = model.predict(obsIdx)

    # Find detrended noise
    return np.std(obs - trend)

def KalmanSnowdepth(series, obs_noise, system_noise=np.diag((1e0,1e-2,1e-3)), outlier_threhold=2e3,
        smooth=False):
    """
    Applies Kalman filter or Kalman smoother to a time series.

    Arguments
    -series: numpy array
        Time series to which to apply filter/smoother
    -obs_noise: float
        Observation noise (ie, observation variance)
    -system noise: 3x3 numpy array
        Noise inherent to the system (ie, noise covariance matrix)
    -outlier_threhold: float
        If the absolute difference between the Kalman filter prediction and true data point is larger than
        this value, the observation is ignored and the update step is not run
    -smooth: boolean
        False: applies Kalman filter. True: applies Kalman smoother.

    Returns:
        If smooth:
            Smoothed mean and covariance matrix at each timestep
        Else:
            Filtered mean and covariance matrix at each timestep
    """
    # Estimate the initial baseline
    baseline = np.median(series[2000:3000][~np.isnan(series[2000:3000])])

    # Label the state parameters. 
    state_params=['depth_1', 'velocity_1', 'baseline_1']

    # First observation that is not nan 
    Y0 = series[np.argwhere(~np.isnan(series))[0]]
    sigma_0 = np.diag((50, 10, 10))
    mu_0 = np.array([-Y0+baseline, 0., baseline]) # Initial state is the first observation 
    dt = .25 # 15 minute intervals.  Velocity in mm/hr

    # Transition Matrix 
    A = np.array([[1, dt, 0], \
    [0,  1, 0], \
    [0,  0, 1]])

    # Control Model 
    B = np.zeros((len(mu_0),len(mu_0)))

    # Observation Matrix
    C = np.array([[-1, 0, +1],]) 

    # Process noise.
    Q = system_noise

    # Observation Noise
    R = obs_noise

    # For now, no control input 
    u = None
    D = None 

    K = Kalman(mu_0, sigma_0, A, B, C, D, Q, R)

    # Filtered mean and covariance (mu_t|t, sigma_t|t)
    sigma_filt = np.zeros((len(series), sigma_0.shape[0], sigma_0.shape[1])) # sigma_0|0, ..., sigma_T|T
    mu_filt = np.zeros((len(series),len(mu_0))) # mu_0|0, ..., mu_T|T
    # Initial values
    mu_filt[0,:] = mu_0
    sigma_filt[0, :, :] = obs_noise

    # Predicted means and covariances (mu_t|t-1, sigma_t|t-1). Indexed by t-1!
    sigma_pred = np.zeros((len(series) - 1, sigma_0.shape[0], sigma_0.shape[1])) # sigma_1|0, ..., sigma_T|T-1
    mu_pred = np.zeros((len(series) - 1, len(mu_0))) # mu_1|0, ..., mu_T|T-1

    # Filtering loop
    for t in range(1, len(series)):
        K.predict()
        # Save mu_t|t-1 and sigma_t|t-1
        sigma_pred[t-1] = K.sigma
        mu_pred[t-1] = K.mu

        # Only update the state if we have a valid measurement 
        # and it is not an obvious outlier (threhold is a change of >2meters)
        difference = np.abs((series[t]-np.dot(K.C, K.mu)))

        if not np.isnan(series[t]) and (difference < 2e3):
            K.update(series[t])

        # Save mu_t|t, sigma_t|t
        mu_filt[t] = K.mu
        sigma_filt[t, :, :] = K.sigma

    if not smooth:
        return mu_filt, sigma_filt
    else:
        # Store smoothed mean and covariance
        sigma_smooth = np.zeros((len(series), sigma_0.shape[0], sigma_0.shape[1])) # sigma_0|T, ..., sigma_T|T
        mu_smooth = np.zeros((len(series),len(mu_0))) # mu_1|1, ..., mu_T|T
        # Initialize smoothed mean and covariance
        mu_smooth[-1] = mu_filt[-1] # mu_T|T = mu_T|T(filtered)
        sigma_smooth[-1] = sigma_filt[-1] # sigma_T|T = sigma_T|T(filtered)

        # Smoothing loop. Runs from t=T-1 to t=0    
        for t in range(len(series)-2, -1, -1): # t = T-1, ..., 0
            # Backward Kalman gain matrix
            J = np.dot(np.dot(sigma_filt[t], K.A.T), np.linalg.pinv(sigma_pred[t]))

            # Smoothed mean
            mu_smooth[t] = mu_filt[t] + np.dot(J, mu_smooth[t+1] - mu_pred[t])

            # Smoothed covariance
            sigma_smooth[t, :, :] = sigma_filt[t] + np.dot(np.dot(J, sigma_smooth[t+1] - sigma_pred[t]), J.T)

        return mu_smooth, sigma_smooth
