import numpy as np
from scipy.stats import linregress
from multiprocessing import pool 
from functools import partial 
import pandas as pd

def alignSeries(YRaw):
    """
    Aligns time series from different sensors

    Arguments
    -YRaw: numpy array
    Time series to be aligned. Rows index the sensors, columns index the observations.

    Returns: numpy array
    Aligned data. Alignment is performed by iteratively regressing each sensors' data
    onto the mean time series.
    """
    Y = YRaw.copy()
    N = Y.shape[0]

    # Keep track of the mean r^2 value
    meanR2 = 0.0
    oldMeanR2 = 0.0

    # Keep track of overall slope and intercept
    cumSlopes = np.zeros(N)    
    cumIntercepts = np.zeros(N)

    for i in range(100):
        slopes = np.zeros(N)
        intercepts = np.zeros(N)

        # Average over sensors
        yAvg = np.nanmean(Y, axis=0)

        # Compute new mean r^2 value
        oldMeanR2 = meanR2
        meanR2 = 0.0

        for s in range(N):
            # Get times where mean and sensor are both not nan
            bothObsIdx = np.where(~(np.isnan(yAvg) + np.isnan(Y[s, :])))[0]
            # Regress
            slopes[s], intercepts[s], rVal, _, _ = linregress(Y[s, bothObsIdx], yAvg[bothObsIdx])

            # Transform the data
            Y[s, :] = slopes[s] * Y[s, :] + intercepts[s]

            # Keep track of the transformation
            if i == 0:
                cumSlopes[s] = slopes[s]
                cumIntercepts[s] = intercepts[s]
            else:
                cumSlopes[s] = cumSlopes[s] * slopes[s]
                cumIntercepts[s] = slopes[s] * cumIntercepts[s] + intercepts[s]

            meanR2 = meanR2 + rVal**2 / float(N)

        # Break if r^2 changed by less than 0.001%
        if np.abs((oldMeanR2 - meanR2) / meanR2) < 0.00001:
            break

    # Demean, standardize
    cumIntercepts = (cumIntercepts - np.nanmean(Y)) / np.nanvar(Y)
    cumSlopes = cumSlopes / np.nanvar(Y)
    Y = (Y - np.nanmean(Y)) / np.nanvar(Y)

    return Y, cumSlopes, cumIntercepts

def scale(arr, axis=0, std=False):
    """
    Version of sklearn.preprocessing.scale that works with data containing nans.

    Arguments
    -arr: numpy array
    Data
    -std: bool
    Indicates whether or not to scale columns to have stdev equal to 1

    Returns: numpy array, numpy array, numpy array
    Copy of arr. Columns are shifted to have mean 0 (and stdev = 1 if std == True).
    Column means. 
    Column stdevs if std == True
    """
    # Get mean for each column
    means = np.nanmean(arr, axis=axis)#, keepdims=True)

    # Check if any of the data series are all nans
    ####if np.isnan(means).sum() != 0:
    ###    print "Warning: some of the time series are all nan!"

    if not std:
        return arr - means, means
    else:
        # Get stdev for each column
        stds = np.nanstd(arr, axis=axis)
        # If a column's standard deviation is 0, there's no need to rescale the data!
        stds[stds == 0.0] = 1.0

        return (arr - means) / stds, means, stds

def descale(arr, means, stds=None):
    """
    Inverse of scale
    """
    if stds == None:
        return arr + means
    else:
        stdsNoZero = stds.copy()
        stdsNoZero[stdsNoZero == 0.0] = 1.0
        return arr * stdsNoZero + means

def nanMSE(data, imputed):
    """
    Computes MSE by summing (data[i][j] - imputed[i][j])^2 over the observed (non-nan) elements of data.
    """
    e = data - imputed
    e[np.isnan(data)] = 0

    return np.sum(np.square(e))

def obsFrac(data):
    """
    Computes fraction of points which are observed (ie, non-nan) in data.
    """
    return float(data.size - np.isnan(data).sum()) / data.size

def longestMissingSeq(arr):
    """
    Determines longest window over which none of the sensors were taking data.

    Returns: double, double
        Length of the sequence and index of sensor for which it occurred
    """
    curLongest = 0
    curLongestSensor = None
    curSeq = 0

    for i, nans in enumerate(np.isnan(vals).T):
        for p in nans:
            if p:
                curSeq = curSeq + 1

                if curSeq > curLongest:
                    curLongest = curSeq
                    curLongestSensor = i
            else:
                curSeq = 0

    return curLongest, i

def getMidnightNoonIdx(dateTimes, ti, tf):
    ts = [] # times at which it was noon or midnight
    labels = [] # label for the time

    for t in range(ti, tf):
        dt = dateTimes[t]

        if dt.endswith("00:00:00"):
            ts.append(t)
            labels.append(dt[5:-8] + "12am")
        elif dt.endswith("12:00:00"):
            ts.append(t)
            labels.append(dt[5:-8] + "12pm")

    return ts, labels

def ding():
    # Makes a noise!
    import subprocess
    subprocess.call(["echo", "\a"])

def GetMAD_Obs(i, x, window_size):
    '''
    Return the median absolute deviation of an observation i over a moving window. 

    Parameters
    ----------
    i : int
        observation index
    x : 1D array
        observation list
    window_size: int
        how many observations are included on each side of the target observation. 


    Returns
    -------
    MAD : int
        Modified z-score in median absolute deviations.
    '''
    x = np.asarray(x)

    # If our observation is an NaN, just return NaN
    if np.isnan(x[i]):
        return x[i]

    n_obs = x.shape[0]

    start_i = np.max( (0,i - window_size) ) # Don't let start i go negative 
    end_i = np.min( (i + window_size, n_obs) ) # i < n_obs 

    # We also need to remove NaN values to compute the median 
    obs_set = x[start_i:end_i][np.logical_not(np.isnan(x[start_i:end_i]))]

    median = np.median( obs_set )
    MAD = np.median( np.abs((obs_set-median)) )
    mod_z_score = np.abs(x[i]-median)/MAD    

    return mod_z_score

def GetMad(x, window_size, threshold):
    '''
    Return the median absolute deviation of an observation list over a moving window. 

    Parameters
    ----------
    x : 1D array
        observation list
    window_size: int
        how many observations are included on each side of the target observation. 
    threshold: float
        what is the modified z-score threshold for outliers 

    Returns
    -------
    MAD : float array 
        Modified z-score in median absolute deviations.
    '''
    # Multithreaded because, why not?
    p = pool.Pool()
    func = partial(GetMAD_Obs, x=x, window_size=window_size ) 
    mod_z = np.array(p.map(func, range(len(x))))
    p.close()

    new_obs = x.copy()
    new_obs[mod_z>threshold] = np.nan # set outliers to NaN
    return new_obs

def loadAndClean(dataFileName, toDrop=['Unnamed: 0', 'Unnamed: 0.1', 'datetime'], tBL=[2000, 3000],
        prefix="snowdepth_"):
    """
    Loads and cleans snow depth data
    """
    df_raw = pd.read_csv(dataFileName)
    # Drop the junk columns
    df_cleaned = df_raw.drop(toDrop, axis=1)

    for i in range(1, len(df_cleaned.columns) / 2 + 1):
        series = prefix + "%i"%i

        if not series in toDrop:
            # Remove voltage spikes
            df_cleaned[series] = GetMad(df_cleaned[series].values[:], window_size=600,
                                                    threshold=3)

            if not tBL is None:
                # Shift to account for baseline
                baseline = np.nanmean(df_cleaned[series][tBL[0]:tBL[1]])
                df_cleaned[series] = baseline - df_cleaned[series]
    
    df_cleaned.sort_index(axis=1, inplace=True)

    return df_cleaned


