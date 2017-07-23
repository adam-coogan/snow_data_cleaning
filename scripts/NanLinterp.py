import numpy as np
import pandas as pd

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def InterpolateNaNs(y):
    nans, x = nan_helper(y)

    if sum(nans) < y.size:
        new = y.copy()
        new[nans] = np.interp(x(nans), x(~nans), y[~nans])

        return new
    else:
        ###print "Warning: cannot interpolate since array only contains nans. This may break your analysis."
        return y

def getNanLinterp(df_clean, sensorPrefix="snowdepth_", prefixCols=3):
    """
    Linearly interpolates to fill in nans
    """
    """
    df_linterp = df_clean.copy()

    # Works for dataframes
    for s in df_clean.columns:
        if sensorPrefix in s:
            df_linterp[s] = InterpolateNaNs(df_linterp[s])
    """

    df_linterp = df_clean.copy().T

    for i, series in enumerate(df_linterp):
        df_linterp[i] = InterpolateNaNs(series)

    return df_linterp.T


