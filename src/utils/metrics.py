__author__ = "alvrogd"


import math
import typing

import numpy as np


def compute_metrics(targets: typing.List[float], outputs: typing.List[float]) -> typing.List[float]:
    # Based on: https://github.com/Havi-muro/SeBAS_project/blob/d89a4909efde7862b168cd8c8bc2dad130655f86/main.py
    
    x, y = np.array(targets, dtype=np.float32), np.array(outputs, dtype=np.float32)
    n = len(x)

    so, sp   = x.sum(), y.sum()
    so2, sp2 = (x ** 2).sum(), (y ** 2).sum()

    sum2   = ((x - y) ** 2).sum()
    sumabs = np.abs(x - y).sum()

    sumdif = (x - y).sum()
    cross  = (x * y).sum()

    obar, pbar = x.mean(), y.mean()

    sdo, sdp = math.sqrt(so2 / n - obar ** 2), math.sqrt(sp2 / n - pbar ** 2)
    c        = cross / n - obar * pbar
    r        = c / (sdo * sdp)
    r2       = r ** 2
    b        = r * sdp / sdo
    a        = pbar - b * obar
    mse      = sum2 / n
    mae      = sumabs / n

    msea = a ** 2
    msei = 2 * a * (b - 1) * obar
    msep = ((b - 1) ** 2) * so2 / n
    mses = msea + msei + msep
    mseu = mse - mses

    rmse  = math.sqrt(mse)
    # Absolute value to avoid negative RRMSE values when targeting NMDS
    rrmse = rmse / np.abs(obar)
    rmses = math.sqrt(mses)
    rmseu = math.sqrt(mseu)

    pe1 = (np.abs(y - obar) + np.abs(x - obar)).sum()
    pe2 = ((np.abs(y - obar) + np.abs(x - obar)) ** 2).sum()
    d1  = 1 - n * mae / pe1
    d2  = 1 - n * mse / pe2

    zb      = [obar, pbar, sdo, sdp, r, a, b, mae, d1, rmse, rrmse, rmses, rmseu, d2]
    results = [r2, rrmse, rmses, rmseu]
    
    return results
