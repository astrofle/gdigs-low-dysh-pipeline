""" """

import numba as nb
import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit

#####################
# Global parameters #
#####################

# Maximum neighbourhood size
MAX_PIXELS = 8

# smoothing default params
KERNEL_M = 40
KERNEL_N = 20
SIGMA_M = 7.5
SIGMA_N = 15

# dilation default params
STRUCT_SIZE = 3

p = 1.5
m = np.arange(1, MAX_PIXELS)
M = 2 ** (m - 1)


def ArPLS(y, lam=1e4, ratio=0.05, itermax=10):
    """
    copy from https://irfpy.irf.se/projects/ica/_modules/irfpy/ica/baseline.html

    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Inputs:
        y:
            input data (i.e. SED curve)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector
    """

    N = len(y)
    D = sparse.eye(N, format="csc")
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]

    D = D.T
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        Z = W + lam * D.dot(D.T)
        z = sparse.linalg.spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt

    return z


def arpls_mask(data, line_threshold=5):
    """
    Computes a mask to cover the RFI in a data set based on ArPLS-ST.

    Inputs:
        data:
            array containing the signal and RFI
        eta_i:
            List of sensitivities

    Outputs:
        mask:
            the mask covering the identified RFI
    """

    # compute SED curve
    freq_mean = data.mean(axis=1)
    # estimate the baseline of SED curve based on ArPLS
    bl = ArPLS(freq_mean, lam=100000)
    # compute the difference between SED curve and its baseline
    diff = freq_mean - bl
    # compute the first threshold value for band RFI mitigation
    popt = ksigma(diff)
    # band RFI mitigation
    line_mask = _run_sumthreshold_arpls(diff, line_threshold * popt)

    line_index = np.where(line_mask == True)[0]
    final_curve = freq_mean.copy()
    final_curve[line_index] = bl[line_index]
    valid_index = np.where(line_mask == False)[0]
    valid_data = data - final_curve[:, np.newaxis]
    valid_data = valid_data[valid_index]

    # compute the first threshold value for blob RFI mitgation
    popt_point = ksigma(valid_data)
    # blob RFI mitigation
    mask = blob_mitigation(data, final_curve, line_mask, 5 * popt_point)

    mask[line_index] = True

    return mask


def blob_mitigation(data, baseline, line_mask, threshold):
    """
    The function to identify the blob RFI

    Inputs:
        data:
            the input data
        baseline:
            the estimated baseline of the input data
        line_mask:
            the band mask
        threshold:
            the first threshold value of the sumthreshold algorithm
    Outputs:
        blob RFI mask
    """

    valid_index = np.where(line_mask == False)[0]
    valid_data = data - baseline[:, np.newaxis]
    valid_data = valid_data[valid_index]
    point_mask_temp = _run_sumthreshold_arpls(valid_data, chi_1=threshold)
    point_mask = np.full(data.shape, False)
    point_mask[valid_index] = point_mask_temp

    return point_mask


def ksigma(data):
    """
    Find the standard deviation of a Gaussian distribution.

    Inputs:
        data:
            input data
    Output:
        popt:
            the estimated standard deviation of the input data
    """

    med = np.nanmedian(data)
    std = np.nanstd(data)
    hist_result = np.histogram(data, bins="auto", density=True)
    x_val = (hist_result[1][1:] + hist_result[1][:-1]) / 2

    def gauss(x, sigma):
        return np.exp(-((x - med) ** 2.0) / (2.0 * sigma**2.0)) / (
            np.sqrt(2.0 * np.pi) * sigma
        )

    popt, _ = curve_fit(gauss, x_val, hist_result[0], p0=[std])

    return popt


@nb.njit
def _sumthreshold(data, mask, i, chi, ds0, ds1):
    """
    The operation of summing and thresholding.
    copy from https://github.com/cosmo-ethz/seek/blob/master/seek/mitigation/sum_threshold.py

    Input:
        data:
            input data for sumthreshold
        mask:
            original mask
        i:
            number of iterations
        chi:
            thresholding criteria
        ds0:
            dimension of the first axis
        ds1:
            dimension of hte second axis

    Output:
        SumThredshold mask
    """
    tmp_mask = mask[:]
    for x in range(ds0):
        sum = 0.0
        cnt = 0

        for ii in range(i):
            if mask[x, ii] != True:
                sum += data[x, ii]
                cnt += 1

        for y in range(i, ds1):
            if sum > chi * cnt:
                for ii2 in range(i):
                    tmp_mask[x, y - ii2 - 1] = True

            if mask[x, y] != True:
                sum += data[x, y]
                cnt += 1

            if mask[x, y - i] != 1:
                sum -= data[x, y - i]
                cnt -= 1

    return tmp_mask


def _run_sumthreshold_arpls(diff_temp, chi_1=3):
    """
    A function to call sumthreshold for a list of threshold value

    Inputs:
        diff_temp:
            The difference of the data and the estimated baseline
        chi_1:
            The first threshold value
    Output:
        SumThredshold mask

    """
    res = diff_temp.copy()
    # use first threshold value to compute the whole list of threshold for sumthreshold algorithm
    chi_i = chi_1 / p ** np.log2(m)
    if len(res.shape) == 1:
        res = res[:, np.newaxis]
    st_mask = np.full(res.shape, False)

    for mi, chi in zip(M, chi_i):
        if mi == 1:
            st_mask = st_mask | (chi <= res)
        else:
            if diff_temp.shape[-1] != 1:
                st_mask = _sumthreshold(res, st_mask, mi, chi, *res.shape)

            st_mask = _sumthreshold(res.T, st_mask.T, mi, chi, *res.T.shape).T

    return st_mask
