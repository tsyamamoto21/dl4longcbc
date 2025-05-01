#!/usr/bin/env python
import numpy as np
from scipy.signal.windows import tukey
import pickle
from pycbc.waveform import get_td_waveform
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
# from pycbc.psd import interpolate
from pycbc.detector import Detector
from dl4longcbc.utils import adjust_waveform_length


def main(ndata=3, seed=128):

    # Set seed (to be implemented)

    # Data path (to be implemented)

    # Strain parameters
    fs = 4096
    duration = 8.0
    tcoalescence = 7.0
    tlen = int(fs * duration)
    f_lower = 15.0
    tgps = 1192529720

    # Waveform parameters
    approximant = 'IMRPhenomD'
    mcmin, mcmax = 50.0, 200.0
    # etamin, etamax = 0.1, 0.25
    # amin, amax = -0.9, 0.9

    # Generate psd and detectors
    delta_f = 1.0 / duration
    flen = int(fs / delta_f) // 2 + 1
    psd = aLIGOZeroDetHighPower(flen, delta_f, 0.0)
    psd[0] = psd[1]  # Zero is replaced by the neighboring value
    psd[len(psd) - 1] = psd[len(psd) - 2]  # Zero is replaced by the neightboring value

    ifolist = {key: Detector(key) for key in ['H1', 'L1', 'V1']}
    parameterlist = {i: {} for i in range(ndata)}
    strainlist = {key: np.zeros((ndata, 1, tlen), dtype=np.float32) for key in ifolist.keys()}

    for idx in range(ndata):
        # mtot = np.random.uniform(mtotmin, mtotmax)  # M_sun
        mc = np.random.uniform(mcmin, mcmax)
        # q = np.random.uniform(qmin, qmax)
        eta = 0.25
        m1 = mass1_from_mchirp_eta(mc, eta)
        m2 = mass2_from_mchirp_eta(mc, eta)
        # a1 = np.random.uniform(amin, amax)
        # a2 = np.random.uniform(amin, amax)
        a1 = 0.0
        a2 = 0.0
        dec = np.arcsin(np.random.uniform(-1, 1))
        ra = np.random.uniform(0.0, 2.0 * np.pi)
        pol = np.random.uniform(0.0, np.pi)
        inclination = np.arcsin(np.random.uniform(-1.0, 1.0))
        phi0 = np.random.uniform(0.0, 2.0 * np.pi)
        params = {
            'approximant': approximant,
            'mass1': m1,
            'mass2': m2,
            'spin1z': a1,
            'spin2z': a2,
            'delta_t': 1.0 / fs,
            'f_lower': f_lower,
            'inclination': inclination,
            'distance': 400.0,
            'phi0': phi0
        }
        locparams = {
            'declination': dec,
            'right_ascension': ra,
            'polarization': pol,
            't_gps': tgps
        }
        parameterlist[idx] = params | locparams

        # Generate waveforms
        hp, hc = get_td_waveform(**params)
        if approximant == 'SEOBNRv4':
            # Prevent an artificial peak to appear in whitened strain
            w = tukey(len(hp), alpha=1.0 / 8.0)
            w[len(hp) // 2:] = 1.0
            hpj, hcj = adjust_waveform_length(hp * w, hc * w, duration, merger=tcoalescence / duration)
        else:
            hpj, hcj = adjust_waveform_length(hp, hc, duration, merger=tcoalescence / duration)

        # Adjust GPS time
        hpj.start_time = hcj.start_time = tgps - tcoalescence

        for key in ifolist.keys():
            ifo = ifolist[key]
            fp, fc = ifo.antenna_pattern(**locparams)
            td_from_earth_center = ifo.time_delay_from_earth_center(ra, dec, tgps)
            strain = fp * hpj + fc * hcj
            strain.roll(int(td_from_earth_center / hpj.delta_t))
            strain_wh = (strain.to_frequencyseries() / psd ** 0.5).to_timeseries()
            strainlist[key][idx, 0] = strain_wh

        # 



    np.savez('testfile.npz', **strainlist)
    with open('parameter.pkl', 'wb') as f:
        pickle.dump(parameterlist, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process the noise strain')
    parser.add_argument('outdir', type=str, help='Directory name including `train` or `validate` or `test`.')
    parser.add_argument('ndata', type=int, help='Data number')
    args = parser.parse_args()
    main(args.ndata)
