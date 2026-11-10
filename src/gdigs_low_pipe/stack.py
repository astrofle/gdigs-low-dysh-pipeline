""" """

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from dysh.fits import GBTFITSLoad, gbtfitsload
from scipy.stats import normaltest


def stack_auto(
    file_list,
    line_list,
    output,
    ledge=400,
    redge=400,
    discard_threshold=0.1,
    verbose=True,
):
    """ """

    df = pd.read_csv(line_list)

    # Prepare output dir.
    output = Path(output)
    output_path = output.parent
    output_path.mkdir(exist_ok=True, parents=True)

    # Prepare output files.
    shutil.copyfile(file_list[0], output)
    n_file = output.with_suffix(".n.fits")
    shutil.copyfile(file_list[0], n_file)
    cov_file = output.with_suffix(".cov.fits")
    shutil.copyfile(file_list[0], cov_file)

    # Prepare data containers.
    ntime = fits.getval(file_list[0], "NAXIS2", ext=1)
    nchan = int(fits.getval(file_list[0], "NCHAN", ext=0))

    x_axis = np.arange(0, nchan, 1) - nchan / 2

    stack = np.zeros((ntime, nchan), dtype="d")
    weights = np.zeros((ntime, nchan), dtype="d")
    count = np.zeros((ntime, nchan), dtype="l")
    level = np.zeros((ntime, nchan), dtype="d")
    tsys = np.zeros((ntime), dtype="d")
    exposure = np.zeros((ntime), dtype="d")

    for i, f in enumerate(file_list):
        pqn = fits.getval(f, "RRLN", ext=0)
        ifn = fits.getval(f, "IFNUM", ext=0)
        pln = fits.getval(f, "PLNUM", ext=0)

        use_col = df[(df["ifnum"] == ifn) & (df["plnum"] == pln) & (df["n"] == pqn)][
            "use"
        ].values
        if len(use_col) > 0:
            use = use_col[0]
        else:
            use = False

        if not use:
            print(f"Skip {f}")
        else:
            print(f"Processing {f} ({i})")

            sdf = GBTFITSLoad(f)
            data = sdf["DATA"]

            # Mask is 0 for points that won't be used.
            mask = np.isfinite(data)

            if mask.sum() == 0:
                # No valid data.
                print("No valid data.")
                continue
            else:
                print(f"Fraction of valid data: {mask.sum() / mask.size}")

            if mask.sum() / mask.size < discard_threshold:
                print("Discarding.")
                continue

            for j, d in enumerate(data):
                if mask[j].sum() < nchan / 2:
                    mask[j] = 0
                    continue
                x = x_axis[mask[j]]
                y = d[mask[j]]
                pfit = np.polyfit(x, y, 5)
                pval = np.poly1d(pfit)(x_axis)
                data[j] = d - pval
                res = normaltest(data[j][mask[j]])
                mask[j] = (res.pvalue > 0.05) * mask[j]

            if mask.sum() == 0:
                print("No good data left after baseline subtraction.")
                continue
            else:
                print(
                    f"Fraction of valid data after baseline subtraction: {mask.sum() / mask.size}"
                )

            if mask.sum() / mask.size < discard_threshold:
                print("Discarding.")
                continue

            weight = np.empty_like(data)
            weight[:, :] = (
                1.0
                / np.nanvar(np.c_[data[:, :ledge], data[:, -redge:]], axis=1)[
                    :, np.newaxis
                ]
            )
            weight[~mask] = 0.0

            if verbose:
                print(
                    f"Non finite values in weights: {np.sum(~np.isfinite(weight))}/{weight.size}"
                )
                print(f"Mean weight: {np.nanmean(weight[mask]):.3f}")
                # if np.isnan(np.nanmean(weight[mask])):
                #     break
            count += (mask).astype("l")
            stack[mask] += (data * weight)[mask]
            level[mask] += (pqn * weight)[mask]
            weights += weight
            tsys += sdf["TSYS"] * weight[:, 0]
            exposure += sdf["EXPOSURE"] * weight[:, 0]

            del sdf

    # Combine.
    stack /= weights
    level /= weights
    tsys /= weights.mean(axis=1)
    exposure /= weights.mean(axis=1)

    sdf = GBTFITSLoad(file_list[0])
    sdf["TSYS"] = tsys
    sdf["EXPOSURE"] = exposure

    gbtfitsload.HAS_FITSIO = False

    sdf["DATA"] = stack
    sdf.write(output, overwrite=True)

    sdf["DATA"] = count
    sdf.write(cov_file, overwrite=True)

    sdf["DATA"] = level
    sdf.write(n_file, overwrite=True)

    gbtfitsload.HAS_FITSIO = True
