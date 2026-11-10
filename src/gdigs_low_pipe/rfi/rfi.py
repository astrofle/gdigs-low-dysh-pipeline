"""
RFI flagging functions.
"""

import numpy as np
from astropy.stats import sigma_clip


def get_calibrated(sb):
    """
    Returns the calibrated data of `sb`.

    Parameters
    ----------
    sb : `~dysh.spectra.scan.ScanBlock`
    """
    nint = sb.nint
    nchn = sb.nchan
    data = np.ma.empty((nint, nchn), dtype=float)
    c = 0
    for i, s in enumerate(sb.data):
        d = s.calibrated
        data[c : c + len(d)] = d
        c += len(d)
    return data


def set_calibrated_mask(sb, mask):
    """
    Sets the mask of `sb`.

    Parameters
    ----------
    sb : `~dysh.spectra.scan.ScanBlock`
    """
    c = 0
    for i, s in enumerate(sb.data):
        nint = s.nint
        s.calibrated.mask = mask[c : c + nint]
        c += nint


def mask_ref(sb):
    """ """
    data = get_calibrated(sb)
    diff = data - np.roll(data, 1, axis=1)
    scdiff = sigma_clip(diff, stdfunc="mad_std", axis=1, grow=1)
    set_calibrated_mask(sb, scdiff.mask)

    sb_avg = sb.timeaverage()
    diff = sb_avg.data - np.roll(sb_avg.data, 1)
    scdiff = sigma_clip(diff, stdfunc="mad_std", grow=1)
    sb_avg.mask = scdiff.mask

    return sb_avg
