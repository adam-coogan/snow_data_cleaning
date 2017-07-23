import pandas as pd
import numpy as np
from Utilities import *
from LFImputer import LFImputer
from multiprocessing.dummy import Pool
from kalman import KalmanSnowdepth, EstimateObservationNoise
from SSMTools import ssmEM, ssmEMAug
from DFMThirdGen import estimateDFM

def windowObsFrac(data, ws):
    """
    Determines fraction of data that was observed in windows of size ws. Returns the observed fraction from t:t+ws for t in 0, len(data)-ws.
    """
    obs = []
    for t in range(0, len(data) - ws):
        obs.append(obsFrac(data[t:(t+ws), :]))

    return np.asarray(obs)

def removeChunk(vals, ti, ws, sensor, gapSize):
    """
    Removes a chunk of size gapSize from vals[ti:ti+ws, sensor].
    Returns data with chunk removed and list of times at which data was removed.
    """
    removed = vals.copy()

    # Determine times at which values were measured
    knownTimes = np.where(~np.isnan(removed[ti:ti+ws - gapSize, sensor]))[0] + ti

    # Choose time at which to delete known data
    startGap = np.random.choice(knownTimes, 1)[0]
    # Delete the data
    removed[startGap:startGap+gapSize, sensor] = np.nan

    return removed, np.asarray(range(startGap, startGap+gapSize))

def genChunkTestSets(data, nSets, ws, gapSize, dirName="test_data/", ofCut=0.9):
    """
    Generates test data sets. Saves the times set to nan, the time window and the sensor for which values were
    blanked out to a csv. Example:
        genTestSets(df_cleaned.values, 3, 300, dirName="test_data/intervalLen_300")
    """
    # Start times for windows with at least ofCut of data observed
    tOFCut = np.where(windowObsFrac(data, ws) > ofCut)[0]

    # Choose times for test intervals
    np.random.seed(np.random.randint(0, 100))
    sampleTs = np.random.choice(tOFCut, size=nSets, replace=False)

    for ti in sampleTs:
        # Randomly select a sensor
        sensor = np.random.randint(0, data.shape[1])
        # Remove some data to use for testing
        _, removedTimes = removeChunk(data, ti, ws, sensor, gapSize)

        # Save data in csvs
        np.savetxt(dirName + "/ti=%i_tf=%i_sensor=%i.csv"%(ti, ti+ws, sensor), removedTimes, \
                delimiter=" ", fmt="%i")

def removeData(vals, ti, ws, sensor, frac):
    """
    Removes specified fraction of entries from a sensor's time series.

    Arguments
    -vals
    Data array
    -ti, ws, sensor
    Data is removed from vals[ti:ti+ws, sensor]
    -frac
    Fraction of known data to be deleted
    """
    removed = vals.copy()

    # Determine times at which values were measured
    knownTimes = np.where(~np.isnan(removed[ti:ti+ws, sensor]))[0] + ti

    # Choose times at which to delete known data
    removedTimes = np.sort(np.random.choice(knownTimes, int(frac * ws), replace=False))
    # Delete the data
    removed[removedTimes, sensor] = np.nan

    return removed, removedTimes

def genTestSets(data, nIntervals, intervalLen, dirName="test_data/", frac=0.2, ofCut=0.9):
    """
    Generates test data sets. Saves the times set to nan, the time window and the sensor for which values were
    blanked out to a csv. Example:
        genTestSets(df_cleaned.values, 3, 300, dirName="test_data/intervalLen_300")
    """
    # Start times for windows with at least ofCut of data observed
    tOFCut = np.where(windowObsFrac(data, intervalLen) > ofCut)[0]

    # Choose times for test intervals
    np.random.seed(np.random.randint(0, 100))
    sampleTs = np.random.choice(tOFCut, size=nIntervals, replace=False)

    for ti in sampleTs:
        # Randomly select a sensor
        sensor = np.random.randint(0, data.shape[1])
        # Remove some data to use for testing
        _, removedTimes = removeData(data, ti, intervalLen, sensor, frac)

        # Save data in csvs
        np.savetxt(dirName + "/ti=%i_tf=%i_sensor=%i.csv"%(ti, ti+intervalLen, sensor), removedTimes, \
                delimiter=" ", fmt="%i")

def loadTestSets(fullData, dirName="test_data/", tsNames=None):
    """
    Loads test datasets generated with genTestSets

    Returns:
        List of test datasets. Each dataset is a dictionary with keys ti, intervalLen, sensor (the sensor for
        which data was removed), fullData (full, known dataset), data (artificially degraded dataset) and
        removedTimes.
    """
    if tsNames is None:
        import glob
        tsNames = glob.glob(dirName + "*.csv")
    else:
        tsNames = [dirName + tsn for tsn in tsNames]

    def parseTSName(tsn):
        """
        Given a test data set filename, returns ti, interval length and sensor for which data was removed.
        """
        parsedName = [int(nc.split("=")[1]) for nc in tsn.split(".")[0].split("/")[-1].split("_")]
        return {"ti": parsedName[0], "intervalLen": parsedName[1] - parsedName[0], "sensor": parsedName[2]}

    testSets = []

    # Load each test set's list of removed times in turn
    for tsn in tsNames:
        testSet = parseTSName(tsn)

        # Remove data from specified times
        testData = fullData.copy()
        removedTimes = np.loadtxt(tsn, dtype="int")
        testData[removedTimes, testSet["sensor"]] = np.nan

        testSet["fullData"] = fullData
        testSet["data"] = testData
        testSet["removedTimes"] = removedTimes
        testSets.append(testSet)

    return testSets

def genBootstrapData(fullData, dirName="bootstrap_data/", ti=None, tf=None, n=1, blockLen=7):
    """
    Generates bootstrap datasets from the full dataset by dividing the interval [ti, tf) into blocks of length
    blockLen and resampling each block with replacement. This is done for each sensor.
    """
    # If initial (final) time not given, apply block bootstrap to whole data set
    if ti == None:
        ti = 0
    if tf == None:
        tf = fullData.shape[0]

    # Reset seed
    np.random.seed()

    bsSets = []

    for i in range(0, n):
        bsSet = fullData.copy()

        # Loop over the sensors
        for sensor in range(fullData.shape[1]):
            # Loop over the blocks
            for tStart in range(ti, tf, blockLen):
                # Resample only the non-nan datapoints
                # TODO: is this a valid way of doing this???
                oldBlockNonNans = bsSet[tStart:tStart+blockLen, sensor].copy()
                oldBlockNonNans = oldBlockNonNans[np.isfinite(oldBlockNonNans)]

                for t in range(tStart, min(tStart + blockLen, fullData.shape[0])):
                    if not np.isnan(bsSet[t, sensor]):
                        bsSet[t, sensor] = np.random.choice(oldBlockNonNans, 1, replace=False)

        bsSets.append(bsSet)

        # Save the dataset
        np.savetxt(dirName + "/blockLen=%i_%i.csv"%(blockLen, i), bsSet, delimiter=" ", fmt="%f")

    return bsSets

def loadBootstrapData(dirName):
    """
    Simple function that loads bootstrap datasets (or whatever CSVs are in the specified directory).
    """
    import glob
    bsSetNames = glob.glob(dirName + "*.csv")

    bsSets = []

    # Load each bootstrap dataset's CSV
    for bsn in bsSetNames:
        bsSets.append(np.loadtxt(bsn, dtype="float"))

    return bsSets

def stupidImputer(vals, ti, ws):
    """
    Imputes nans by using last observed value for each sensor. This is the stupidest method you could think to
    use.
    """
    imputed = vals.copy()

    for s in range(0, vals.shape[1]):
        for t in range(ti, ti + ws):
            if np.isnan(imputed[t, s]):
                lastObsT = t-1
                while np.isnan(imputed[lastObsT, s]):
                    lastObsT = lastObsT - 1

                imputed[t, s] = imputed[lastObsT, s]

    return imputed

def lfiTestMSEHelper(ts, windowsize=5, beta=2.0):
    """
    This helper method can be passed to testMSE() to find the latent factor filter's test MSE.
    
    Arguments
    -ts: dict
        Keys: "data": full data array, "ti": start time for test data, "intervalLen": length of test datasets
        
    Returns: full imputed data array
    """
    # Run LF imputation algorithm
    lfi = LFImputer(data=ts["data"].copy(), maxSteps=100, beta=beta, nLF=1, alpha=1e-2)
    lfi.filterImpute(ts["ti"], ts["ti"]+ts["intervalLen"], windowSize=windowsize, iters=10)

    return lfi.imputed

def lfiSmootherTestMSEHelper(ts, windowsize=5, beta=2.0):
    """
    This helper method can be passed to testMSE() to find the latent factor filter's test MSE.
    
    Arguments
    -ts: dict
        Keys: "data": full data array, "ti": start time for test data, "intervalLen": length of test datasets
        
    Returns: full imputed data array

    TODO: currently does not work!
    """
    ti = ts["ti"]
    tf = ti + ts["intervalLen"]

    # Run LF imputation algorithm forwards
    lfiF = LFImputer(data=ts["data"], maxSteps=100, beta=beta, nLF=1, alpha=1e-2)
    lfiF.filterImpute(ti, tf, windowSize=windowsize, iters=10)

    # Run LF imputation algorithm backwards. Reverse the relevant part of the array: only need the latest
    # entries for this pass!
    ts["data"][ti:tf+windowsize, :]
    lfiB = LFImputer(data=ts["data"][tf+windowsize-2:ti-1:-1, :].copy(), maxSteps=100, beta=beta, nLF=1,
            alpha=1e-2)

    lfiB = LFImputer(data=ts["data"][tf+windowsize-1:ti-1:-1, :].copy(), maxSteps=100, beta=beta, nLF=1,
            alpha=1e-2)

    lfiB.filterImpute(windowsize, tf-ti+windowsize, windowSize=windowsize, iters=10)

    # Average results over window of interest
    imputed = ts["data"].copy()
    imputed[ti:tf, :] = 0.5 * (lfiF.imputed[ti:tf, :] + lfiB.imputed[::-1][windowsize-1:tf-ti+windowsize-1])

    return imputed

def stupidImputerTestMSEHelper(ts):
    """
    This helper method can be passed to testMSE() to find the stupid imputer's test MSE.
    
    Arguments
    -ts: dict
        Keys: "data": full data array, "ti": start time for test data, "intervalLen": length of test datasets

    Returns: full imputed data array
    """
    return stupidImputer(ts["data"], ts["ti"], ts["intervalLen"])

def zeshiImputerTestMSEHelper(ts, n_pc, max_Iter=100):
    from ZeshiImputer import dineof

    imputed = ts["data"].copy()

    # Split test interval into days
    ti = ts["ti"]
    tf = ti + ts["intervalLen"]
    dayLen = 100 # number of observations per day

    deltaT = 0
    while ti + deltaT < tf:
        # Impute data for the current day
        data = imputed[ti + deltaT:min(ti + deltaT + dayLen, tf), :]
        imputed[ti + deltaT:min(ti + deltaT + dayLen, tf), :] = dineof(data, n_max=n_pc, max_Iter=max_Iter)

        deltaT = deltaT + dayLen

    return imputed

def dineofTestMSEHelper(ts, max_eof=None, max_Iter=1000):
    from DINEOF import DINEOF

    imputed = ts["data"].copy()

    # Split test interval into days
    ti = ts["ti"]
    tf = ti + ts["intervalLen"]
    dayLen = 100 # number of observations per day

    deltaT = 0
    while ti + deltaT < tf:
        # Impute data for the current day
        data = imputed[ti + deltaT:min(ti + deltaT + dayLen, tf), :]

        imputed[ti + deltaT:min(ti + deltaT + dayLen, tf), :] = DINEOF(data)

        deltaT = deltaT + dayLen

    return imputed

def kalmanTestMSEHelper(ts, smooth=False):
    # Nothing special here: just filter the sensor's time series
    sensor = ts["sensor"]
    # Select the sensor's time series. Only should filter/smooth from t=0 to t=ti+intervalLen!
    series = ts["data"][0:ts["ti"]+ts["intervalLen"], sensor]

    # Apply Kalman filter/smoother
    obs_noise = 0.01 * EstimateObservationNoise(series)
    imputedSeries = KalmanSnowdepth(series, obs_noise, system_noise=np.diag((1e0,1e-2,1e-3)), \
            outlier_threhold=2e3, smooth=smooth)[0]

    # Subtract depths from baseline to get physical depth measurements
    imputedSeries = imputedSeries[:, 2] - imputedSeries[:, 0]

    # Return copy with sensor's data imputed
    imputed = ts["data"].copy()
    imputed[0:ts["ti"]+ts["intervalLen"], sensor] = imputedSeries

    return imputed

def ssmEMTestMSEHelper(ts, nLF, maxIt=200):
    """
    Tests how well SSM EM works when run ONLY on the relevant window of data
    """
    YRaw = ts["data"][ts["ti"]:ts["ti"]+ts["intervalLen"], :].T
    # Preprocess
    Y, cs, ci = alignSeries(YRaw)
    XHat, _, _, C, _, _, _, _ = ssmEM(Y, nLF, maxIt=maxIt)

    # Estimate observations
    YHat = np.dot(C, XHat)

    # Descale imputed observations
    for s in range(YRaw.shape[0]):
            YHat[s, :] = (YHat[s, :] - ci[s]) / cs[s]

    # Put sensor's imputed data back into the array
    imputed = ts["data"].copy()
    imputed[ts["ti"]:ts["ti"]+ts["intervalLen"], :] = YHat.T

    return imputed

def singleSensorSSMEMTestMSEHelper(ts, maxIt=100):
    """
    Tests how well SSM EM works when run ONLY on the relevant window of data
    """
    YRaw = ts["data"][ts["ti"]:ts["ti"]+ts["intervalLen"], ts["sensor"]:ts["sensor"]+1].T
    # Preprocess
    XHat, _, _, C, _, _, _, _ = ssmEMAug(YRaw, 1, maxIt=maxIt)

    # Estimate observations
    YHat = np.dot(C, XHat)

    # Put sensor's imputed data back into the array
    imputed = ts["data"].copy()
    imputed[ts["ti"]:ts["ti"]+ts["intervalLen"], ts["sensor"]:ts["sensor"]+1] = YHat.T

    return imputed

def dfmThirdGenTestMSEHelper(ts, nLF, maxIt=10):
    """
    Tests how well a third generation DFM estimator with hidden state augmented by its derivatives works
    """
    YRaw = ts["data"][ts["ti"]:ts["ti"]+ts["intervalLen"], :].T
    # Preprocess
    Y, cs, ci = alignSeries(YRaw)

    # Run DFM
    CHat, XHat = estimateDFM(Y, nLF)

    # Estimate observations
    YHat = np.dot(CHat, XHat)

    # Descale imputed observations
    for s in range(YRaw.shape[0]):
        YHat[s, :] = (YHat[s, :] - ci[s]) / cs[s]

    # Put sensor's imputed data back into the array
    imputed = ts["data"].copy()
    imputed[ts["ti"]:ts["ti"]+ts["intervalLen"], :] = YHat.T

    return imputed

def testMSE(imputeFn, testSets): #dirName="test_data/"):
    """
    Computes test MSE for an imputation method

    Arguments
    -imputeFn
        Function taking test dataset as input that outputs estimates of the missing observations
    -testSets
        Test datasets
    """
    #testSets = loadTestSets(fullData, dirName)

    def impHelper(ts):
        # Run imputation algorithm
        testImputed = imputeFn(ts)

        # Compute squared error, number of points removed
        return [np.square(testImputed[ts["removedTimes"], ts["sensor"]] - ts["fullData"][ts["removedTimes"],
            ts["sensor"]]).sum(), len(ts["removedTimes"])]

    # Parallelize
    pool = Pool(3)
    ses, nPtsRemoveds = zip(*pool.map(impHelper, testSets))

    return sum(ses) / float(sum(nPtsRemoveds))


