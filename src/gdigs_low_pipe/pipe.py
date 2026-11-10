""" """

from pathlib import Path

import bottleneck as bn
import numpy as np
import pandas as pd
import pymupdf
from astropy import units as u
from astropy.io import fits
from dysh.fits import GBTFITSLoad
from dysh.log import logger
from numpy.lib.stride_tricks import sliding_window_view
from rrlpy.freq import find_lines_sb

from gdigs_low_pipe import dq
from gdigs_low_pipe.rfi import arpls
from gdigs_low_pipe.stack import stack_auto


def main(
    data_path,
    projid,
    output_path,
    fluxcal="3C353",
    flags=None,
    sources=None,
    ifnums=None,
    plnums=None,
    flag_rfi=True,
    data_quality=True,
    stack=False,
):
    """ """

    filename = Path(data_path) / f"{projid}/{projid}.raw.vegas"
    outputdir = Path(output_path) / projid

    outputdir.mkdir(exist_ok=True, parents=True)

    w = 64

    vmin = -300
    vmax = 300
    dv = 0.5
    vel_axis = np.arange(vmin, vmax + dv, dv)

    sdf = GBTFITSLoad(filename)

    summary = sdf.get_summary()

    fluxcal_ = summary.SCAN[summary.OBJECT == fluxcal]
    if len(fluxcal_) == 0:
        logger.error(f"No fluxcal for {projid}")
        return
    if summary.loc[fluxcal_.index].PROC.unique() == "OffOn":
        tcal_scan = fluxcal_[1]
        tcal_ref = fluxcal_[0]
    else:
        tcal_scan = fluxcal_[0]
        tcal_ref = fluxcal_[1]

    if sources is None:
        sources = sdf.udata("OBJECT")
        sources.remove(fluxcal)
    if ifnums is None:
        ifnums = sdf.udata("IFNUM")
    if plnums is None:
        plnums = sdf.udata("PLNUM")
    nchan = sdf.nchan()
    chrng = slice(int(nchan * 0.08), int(nchan * 0.92))
    x = np.arange(-nchan // 2, nchan // 2, 1)

    dq_obj = dq.DataQuality(data_quality)
    dq_out = {k: {0: [], 1: []} for k in sources}

    line_list = {k: {"n": [], "ifnum": [], "plnum": [], "use": []} for k in sources}

    sdf.clear_selection()
    for ifnum in ifnums[:]:
        for plnum in plnums[:]:
            logger.info(f"{ifnum=} {plnum=}")

            base_out_path = outputdir / filename.name.replace(
                "raw.vegas", f"ifnum_{ifnum}_plnum_{plnum}"
            )

            tcal = sdf.gettcal(
                scan=int(tcal_scan),
                ref=int(tcal_ref),
                ifnum=ifnum,
                plnum=plnum,
                fdnum=0,
                object=fluxcal,
                zenith_opacity=0.0,
            )

            tcal_roll = np.roll(bn.move_median(tcal.flux, 512), -512 // 2)
            mask = np.isnan(tcal_roll) | np.isnan(tcal.data) | tcal.mask
            tcal_pfit = np.polyfit(x[~mask][chrng], tcal_roll[~mask][chrng], 1)
            tcal_pval = np.poly1d(tcal_pfit)(x)

            logger.info(f"Mean Tcal value: {tcal_pval.mean()} K")

            dq_obj.plot(base_out_path.as_posix() + "_tcal.pdf", tcal)

            for source in sources:
                if source == fluxcal:
                    continue

                logger.info(f"{source=} {ifnum=} {plnum=}")

                outdir = outputdir / source
                outdir.mkdir(exist_ok=True, parents=True)
                source_out_path = outdir / filename.name.replace(  # noqa: F841
                    "raw.vegas", f"ifnum_{ifnum}_plnum_{plnum}"
                )

                sdf.select(
                    object=source,
                    proc=["RALongMap", "DecLatMap"],
                    ifnum=ifnum,
                    plnum=plnum,
                    fdnum=0,
                )
                df = sdf.get_summary(selected=True)
                scans = df.SCAN.unique()
                _scan = int(df["SCAN"].median())
                _intnum = df[df["SCAN"] == _scan]["# INT"].iloc[0] // 2
                sdf.clear_selection()

                logger.info(f"Using scan={_scan} to get RRL locations")
                spec = sdf.gettp(
                    ifnum=ifnum, plnum=plnum, fdnum=0, intnum=_intnum, scan=_scan
                ).timeaverage()
                freq = spec.spectral_axis.to("MHz").value

                qns, frqs = find_lines_sb([freq.min(), freq.max()], line="RRL_HIalpha")

                sb_on = sdf.gettp(
                    ifnum=ifnum, plnum=plnum, fdnum=0, scan=list(scans), cal=True
                )
                sb_of = sdf.gettp(
                    ifnum=ifnum, plnum=plnum, fdnum=0, scan=list(scans), cal=False
                )

                sb_on_data = np.ma.empty((sb_on.nint, sb_on.nchan), dtype=float)
                sb_of_data = np.ma.empty((sb_of.nint, sb_on.nchan), dtype=float)
                exp_on = np.empty((sb_on.nint), dtype=float)
                exp_of = np.empty((sb_of.nint), dtype=float)

                i, j = 0, 0
                for son, sof in zip(sb_on, sb_of):
                    sb_on_data[i : i + son.nint, :] = son.calibrated
                    sb_of_data[j : j + sof.nint, :] = sof.calibrated
                    exp_on[i : i + son.nint] = son.exposure
                    exp_of[j : j + son.nint] = sof.exposure

                    i += son.nint
                    j += sof.nint

                gtcal = np.ma.masked_invalid(sb_on_data - sb_of_data)
                exp = exp_on + exp_of
                gtcal_avg = np.ma.average(gtcal, axis=0, weights=exp)
                gtcal_poly = np.polyfit(x[chrng], gtcal_avg[chrng], 1)
                gtcal_pval = np.poly1d(gtcal_poly)(x)

                cal_on = sb_on_data / gtcal_pval * tcal_pval - tcal_pval
                cal_of = sb_of_data / gtcal_pval * tcal_pval

                if flag_rfi:
                    bl = arpls.ArPLS(
                        cal_on.mean(axis=0), lam=1e8, ratio=0.0, itermax=1000
                    )
                    cal_on_bl = cal_on - bl[np.newaxis, :]
                    v = sliding_window_view(cal_on_bl, (w), axis=1)
                    rms = np.roll(v.std(axis=-1), w, axis=1)
                    mask = (
                        rms
                        > (np.nanmedian(rms, axis=1) + np.nanstd(rms, axis=1))[
                            :, np.newaxis
                        ]
                    )
                    md = np.ma.masked_where(mask, cal_on_bl[:, : nchan - (w - 1)])
                    xp = x[: nchan - (w - 1)]
                    coeffs = np.apply_along_axis(
                        lambda y: np.polyfit(xp[~y.mask], y[~y.mask], 2),  # noqa: B023
                        axis=1,
                        arr=md,
                    )
                    polyv = np.apply_along_axis(
                        lambda c: np.poly1d(c)(x), axis=1, arr=coeffs
                    )
                    cal_on_bl -= polyv
                    z_norm_data = cal_on_bl[:, : nchan - (w - 1)].filled(0) / rms
                    _rfi_mask = arpls.arpls_mask(z_norm_data[:, :].T).T
                    rfi_mask = np.ma.ones_like(cal_on, dtype=bool)
                    rfi_mask[:, : nchan - (w - 1)] = _rfi_mask
                    cal_on = np.ma.MaskedArray(cal_on, mask=rfi_mask)

                    bl = arpls.ArPLS(
                        cal_of.mean(axis=0), lam=1e8, ratio=0.0, itermax=1000
                    )
                    cal_of_bl = cal_of - bl[np.newaxis, :]
                    v = sliding_window_view(cal_of_bl, (w), axis=1)
                    rms = np.roll(v.std(axis=-1), w, axis=1)
                    mask = (
                        rms
                        > (np.nanmedian(rms, axis=1) + np.nanstd(rms, axis=1))[
                            :, np.newaxis
                        ]
                    )
                    md = np.ma.masked_where(mask, cal_of_bl[:, : nchan - (w - 1)])
                    xp = x[: nchan - (w - 1)]
                    coeffs = np.apply_along_axis(
                        lambda y: np.polyfit(xp[~y.mask], y[~y.mask], 2),  # noqa: B023
                        axis=1,
                        arr=md,
                    )
                    polyv = np.apply_along_axis(
                        lambda c: np.poly1d(c)(x), axis=1, arr=coeffs
                    )
                    cal_of_bl -= polyv
                    z_norm_data = cal_of_bl[:, : nchan - (w - 1)].filled(0) / rms
                    _rfi_mask = arpls.arpls_mask(z_norm_data[:, :].T).T
                    rfi_mask = np.ma.ones_like(cal_of, dtype=bool)
                    rfi_mask[:, : nchan - (w - 1)] = _rfi_mask
                    cal_of = np.ma.MaskedArray(cal_of, mask=rfi_mask)

                cal_on -= np.nanmedian(cal_on.filled(np.nan), axis=-1)[:, np.newaxis]
                cal_of -= np.nanmedian(cal_of.filled(np.nan), axis=-1)[:, np.newaxis]

                # Replace data.
                i, j = 0, 0
                for son, sof in zip(sb_on, sb_of):
                    son._calibrated = cal_on[i : i + son.nint, :]
                    sof._calibrated = cal_of[j : j + sof.nint, :]

                    i += son.nint
                    j += sof.nint

                # Split lines and interpolate.
                for n, f in zip(qns, frqs):
                    sb_on_out = sdf.gettp(
                        ifnum=ifnum, plnum=plnum, fdnum=0, scan=list(scans), cal=True
                    )
                    sb_of_out = sdf.gettp(
                        ifnum=ifnum, plnum=plnum, fdnum=0, scan=list(scans), cal=False
                    )

                    use = 1

                    logger.info(f"{source=} {ifnum=} {plnum=} {n=}")

                    spec.rest_value = f * u.MHz
                    velo = spec.velocity_axis_to(
                        unit="km/s", toframe="lsrk", doppler_convention="radio"
                    )
                    low = np.argmin(abs(velo - -310 * u.km / u.s))
                    upp = np.argmin(abs(velo - 310 * u.km / u.s))
                    low, upp = np.sort([low, upp])

                    # Do not stack lines at the bandpass roll off.
                    if upp < chrng.start or low > chrng.stop:
                        use = 0

                    for sn, (son, sof) in enumerate(zip(sb_on, sb_of)):
                        son_out = sb_on_out[sn]
                        sof_out = sb_of_out[sn]

                        logger.info(f"{source=} {ifnum=} {plnum=} {n=} {son=}")

                        if velo[0] - velo[1] < 0:
                            r = -1
                        else:
                            r = 1

                        data = son.calibrated[:, low:upp][:, ::r]
                        idata = np.ma.empty((len(data), len(vel_axis)), dtype=float)
                        idata.mask = np.zeros(idata.shape, dtype=bool)
                        for i, d in enumerate(data):
                            if np.all(d.mask):
                                idata[i] = np.nan
                                idata.mask[i] = True
                            else:
                                idata[i] = np.interp(vel_axis, velo.value[low:upp], d)
                                im = (
                                    np.interp(vel_axis, velo.value[low:upp], d.mask) > 0
                                )
                                idata.mask[i] = im
                        son_out._calibrated = idata[:, ::r]

                        data = sof.calibrated[:, low:upp][:, ::r]
                        idata = np.ma.empty((len(data), len(vel_axis)), dtype=float)
                        idata.mask = np.zeros(idata.shape, dtype=bool)
                        for i, d in enumerate(data):
                            if np.all(d.mask):
                                idata[i] = np.nan
                                idata.mask[i] = True
                            else:
                                idata[i] = np.interp(vel_axis, velo.value[low:upp], d)
                                im = (
                                    np.interp(vel_axis, velo.value[low:upp], d.mask) > 0
                                )
                                idata.mask[i] = im
                        sof_out._calibrated = idata[:, ::r]

                        crval1 = vel_axis[0] * 1e3
                        cdelt1 = dv * 1e3

                        for d in son_out._meta:
                            d["CRVAL1"] = crval1
                            d["CDELT1"] = cdelt1
                            d["CUNIT1"] = "m/s"
                            d["CRPIX1"] = 1
                            d["CTYPE1"] = "VRAD"
                            d["VFRAME"] = 0.0
                        for d in sof_out._meta:
                            d["CRVAL1"] = crval1
                            d["CDELT1"] = cdelt1
                            d["CUNIT1"] = "m/s"
                            d["CRPIX1"] = 1
                            d["CTYPE1"] = "VRAD"
                            d["VFRAME"] = 0.0

                    sb_on_out.extend(sb_of_out)

                    # try:
                    #    sb_avg = sb_on_out.timeaverage(use_wcs=False)
                    # except ZeroDivisionError:
                    #    logger.warning("No data left to average.")
                    #    continue

                    output_fnm = outdir / filename.name.replace(
                        "raw.vegas",
                        f"ifnum_{ifnum}_plnum_{plnum}_n_{n:.0f}.cal.vegas.fits",
                    )
                    sb_on_out.write(output_fnm, overwrite=True)

                    # sb_avg = sb_on_out.timeaverage(use_wcs=False)

                    # mfrac = sb_avg.mask.sum()/len(sb_avg.data)
                    # logger.info(f"Flagged data in SB average: {mfrac*100} %")
                    # if mfrac > 0.2:
                    #    use = 0

                    line_list[source]["n"].append(n)
                    line_list[source]["ifnum"].append(ifnum)
                    line_list[source]["plnum"].append(plnum)
                    line_list[source]["use"].append(use)

                    # dq_obj.plot(source_out_path.as_posix() + f"_n_{n:.0f}_avg.pdf", sb_avg, **{"xaxis_unit": "pix", "title": f"{source} {n=}"})
                    # dq_out[source][1].append(source_out_path.as_posix() + f"_n_{n:.0f}_avg.pdf")

                    fits.setval(output_fnm, "IFNUM", value=ifnum)
                    fits.setval(output_fnm, "PLNUM", value=plnum)
                    fits.setval(output_fnm, "RRLN", value=int(n))
                    fits.setval(output_fnm, "RESTFRQ", value=f * 1e6)
                    fits.setval(output_fnm, "NCHAN", value=len(vel_axis))
                    fits.setval(output_fnm, "USTACK", value=use)

                # Clean up /tmp
                for p in Path("/tmp").glob("*.flags"):
                    p.unlink()

    # Write line lists.
    for s in sources:
        df = pd.DataFrame(line_list[s])
        df.to_csv(outputdir / f"{s}.csv")

    # Merge pdfs.
    if data_quality:
        for s in sources:
            for i in range(2):
                files = dq_out[s][i]
                if len(files) == 0:
                    print(f"Skipping combined plot for {i}")
                    continue
                file0 = pymupdf.open(files[0])
                for j in range(1, len(files)):
                    filej = pymupdf.open(files[j])
                    file0.insert_file(filej)  # merge the docs
                    filej.close()
                    Path(files[j]).unlink()
                file0.save(outputdir / f"{source}_{i}_summary.pdf")
                file0.close()

    # Stack.
    if stack:
        for s in sources:
            outdir = outputdir / s
            file_list = list(
                outdir.glob(f"{projid}.ifnum_*_plnum_*_n_*.cal.vegas.fits")
            )
            line_list = outputdir / f"{s}.csv"
            output = outdir / "stack-auto.fits"
            stack_auto(file_list, line_list, output, ledge=400, redge=400, verbose=True)
