import numpy as np
from NanLinterp import getNanLinterp, InterpolateNaNs
from Utilities import scale, descale

"""
Class for performing latent factor filtering on different segments of a dataset.

> Don't re-impute data if it's already been imputed with the current setting!

_lfGradDesc()
>   -Figure out how to choose a good step size. Needs to vary with the amount of missing data...
    -Implement backtracking line search
    -Implement stopping condition based on MSE?

_lfDecomp()
    -Doesn't handle numLF = 1...
    -Print warning when data for a sensor or at a given time is all nans.
    -Sound an alarm when data is all nans

lfFilterImpute()
>   -Figure out correct boundary conditions...
    -For the love of god use multithreading!!!
    -Write description of algorithm!
    -Think through the theory:
    * Should previously imputed values be included in the fit? I don't think so. Doesn't make sense and won't work online.
    * Could easily make this an argument to this method.
    -Figure out how to handle the cold-start problem (ie, how does this work when you just have one data point?)
    * Related: same problem occurs if all series are constant.
"""

class LFImputer:
    def __init__(self, data, nLF=2, beta=1e4, maxSteps=100, alpha=8e-7):
        """
        Arguments
        -data: numpy array
            Dataframe with column names of the form "sensor_i", where i is an integer
        -nLF: int > 0
            Number of latent factors
        -maxSteps: int > 0
            Maximum number of steps for gradient descent
        -alpha: float > 0
            Gradient descent step size
        -beta: float >= 0
            Regularization parameter
        """
        self._data = data.copy() # sets self.imputed
        self.imputed = data.copy()
        self.lin = getNanLinterp(self.data)

        # Latent factor model parameters
        self.nLF = nLF
        self.beta = beta
        self.maxSteps = maxSteps
        self.alpha = alpha

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d.copy()
        self.imputed = d.copy()

    def _lfGradDesc(self, vals, uI, vI):
        """
        Determines latent factor decomposition for given matrix using gradient descent.

        Arguments:
        -vals: numpy array
            Matrix to decompose of dimension n x m
        -uI: numpy array
            Initial guess for n x K latent factor matrix
        -vI: numpy array
            Initial guess for m x K latent factor matrix

        Returns:
            Fitted values for P and Q.
        """
        if vals.shape[0] <= 1 or vals.shape[1] <= 1:
            print "Should not perform LF decomposition on a vector"
            return None
        elif self.nLF > vals.shape[0] or self.nLF > vals.shape[1]:
            print "Number of latent factors cannot be larger than the rank of the data matrix!"
            return None
        else:
            u = uI.copy()
            v = vI.copy().T

            for step in xrange(self.maxSteps):
                # Current error matrix. Set error equal to zero at missing data indices so they are not
                # included in the gradient descent update.
                e = vals - np.dot(u, v)
                e[np.isnan(vals)] = 0.0

                # Update latent factors, making sure not to change u before updating v
                uNext = u + self.alpha * (2.0 * np.dot(e, v.T) - self.beta * u)
                v = v + self.alpha * (2.0 * np.dot(u.T, e) - self.beta * v)
                u = uNext

            return u, v.T

    def _lfDecomp(self, ti, tf):
        """
        Performs latent factor decomposition using linearly interpolated data for initialization. The
        decomposition is performed after shifting the data to have zero mean.

        Arguments
        -ti, tf: int
            Indices of start and end time of interval in which to perform the LF decomposition. tf - ti must
            be greater than 1. Note that as usual in python the time interval is [ti, tf).

        Returns: numpy array of floats
            Numpy array of same dimensions as data containing known and imputed values.
        """
        # Check whether all data is known already
        if np.isnan(self.data[ti:tf, :]).sum() == 0:
            return self.data[ti:tf, :]
        else:
            vals = self.data[ti:tf, :].copy() 

            # If a sensor only has nans, use last imputed (or observed) value. TODO: not sure what to do here.
            for s in range(vals.shape[1]):
                if np.isnan(vals[:, s]).sum() == tf-ti:
                    # Loop until the last imputed/observed value is found
                    lastObsT = ti
                    while np.isnan(self.imputed[lastObsT, s]) and lastObsT >= 0:
                        lastObsT = lastObsT - 1

                    if lastObsT >= 0:
                        vals[0, s] = self.imputed[lastObsT, s].copy()
                    else:
                        # No data was observed earlier for this sensor. Best we can do is substitute the
                        # average from the other sensors at first time in the interval
                        vals[0, s] = np.nanmean(vals[0, :])

                        # If that average is zero, substitute average over whole data matrix
                        if np.isnan(vals[0, s]):
                            vals[0, s] = np.nanmean(vals)

            # Standardize the columns
            vals, means, stdsNoZero = scale(vals, std=True)

            # Dimensions for constructing latent factor matrices
            n = len(vals)
            m = len(vals[0])

            # Since columns are standardized, draw guesses from N(0, 1)
            uI = np.random.multivariate_normal(np.zeros(vals.shape[0]),
                    np.diag(np.ones(vals.shape[0])), self.nLF).T
            vI = np.random.multivariate_normal(np.zeros(vals.shape[1]),
                    np.diag(np.ones(vals.shape[1])), self.nLF).T

            # Run LF algorithm
            uLF, vLF = self._lfGradDesc(vals, uI, vI)

            # Impute and descale result
            return descale(np.dot(uLF, vLF.T), means, stdsNoZero)

    def filterImpute(self, ti, tf, windowSize, iters=10):
        """
        Imputes missing data using a sliding window. This can be used online (ie, as a filter).

        Arguments
        -ti, tf: int
            Indices of start and end time of interval over which to apply the LF filter. tf - ti must be
            greater than 1. Note that as usual in python the time interval is [ti, tf), so data[tf, :] is not
            imputed.
        -windowSize: int
            Size of time window to use for imputation.

        Returns: numpy array
            Numpy array containing known and imputed values. The array shape is
                tf-ti - nLF, data.shape[1].
            This is because the first time where imputation is valid is ti + nLF: otherwise the data matrix's
            rank is smaller than nLF. If there are no nans, data is returned immediately. 
        """
        imputedVals = np.zeros((tf-ti, self.data.shape[1]))
        
        for deltaT in range(0, tf-ti):
            if ti + deltaT - windowSize >= 0:
                # Run LF imputation multiple times to average over initializations
                for i in range(iters):
                    # Impute ONLY at this time: take last row from imputed data array and put it into next row in
                    # imputed
                    imp = self._lfDecomp(ti + deltaT - windowSize + 1, ti + deltaT + 1)
                    imputedVals[deltaT, :] = imputedVals[deltaT, :] \
                            + imp[-1, :] / (0.0 + iters)

                # Save imputed values
                self.imputed[ti + deltaT, :] = imputedVals[deltaT, :].copy()

                # Should never get here:
                if np.isnan(self.imputed[ti + deltaT, :]).sum() != 0:
                    print "WARNING: imputed nan. Probably caused by using too large a step size alpha."
            else:
                print "Indexing error: need to fix this"

        return imputedVals


