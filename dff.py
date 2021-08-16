import numpy as np


def interp1d_(x, y, x_new):
    from scipy.interpolate import interp1d, pchip_interpolate
    # return interp1d(x,y,kind='cubic')(x_new)
    return pchip_interpolate(x, y, x_new)


def get_baseline_dff(fmean, fneuropil, cont_ratio=0.7, win_=3000, q=0.1):
    import pandas as pd
    fmean_comp = fmean-fneuropil*cont_ratio
    if fmean_comp.min()<0:
        fmean_comp = fmean_comp-fmean_comp.min()+100
    baseline = pd.Series(fmean_comp).rolling(win_, min_periods=1, center=True).quantile(q, interpolation='lower')
    return fmean_comp, baseline, fmean_comp/baseline-1
