#!/usr/bin/env python
import numpy as np
import pickle
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy.units import Mpc
from pycbc.waveform import get_td_waveform
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
from pycbc.filter import highpass, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.detector import Detector
import pycbc.noise
from dl4longcbc.sampler import rejection_sampling
from dl4longcbc.utils import adjust_waveform_length, if_not_exist_makedir


def main(args):
    # Data path
    outdir = args.outdir
    if_not_exist_makedir(outdir)
    ndata = args.ndata

    # Strain parameters
    ifo_name = 'H1'
    ifo = Detector(ifo_name)
    fs = 4096
    delta_t = 1.0 / fs
    total_duration = 32.0
    # duration = 4.0
    tcoalescence = total_duration / 2
    tlen = int(fs * total_duration)
    f_lower = 15.0
    tgps = 1192529720

    # PSD parameters
    delta_f_psd = 1.0 / 16
    psdparam = {
        'length': int(fs / 2 / delta_f_psd) + 1,
        'delta_f': delta_f_psd,
        'low_freq_cutoff': 5.0
    }
    psd_analytic = aLIGOZeroDetHighPower(**psdparam)

    # Waveform parameters
    approximant = 'IMRPhenomD'
    mcmin, mcmax = 50.0, 200.0
    # etamin, etamax = 0.1, 0.25
    # amin, amax = -0.9, 0.9
    dlmin, dlmax = 0.3, 2.0

    def pdf_dl(x):
        return 3.0 * x**2 / (dlmax**3 - dlmin**3)

    # Generate psd and detectors
    # delta_f = 1.0 / duration
    # flen = int(fs / delta_f) // 2 + 1
    # psd = aLIGOZeroDetHighPower(flen, delta_f, 0.0)
    # psd[0] = psd[1]  # Zero is replaced by the neighboring value
    # psd[len(psd) - 1] = psd[len(psd) - 2]  # Zero is replaced by the neightboring value

    # parameterlist = {i: {} for i in range(ndata)}
    # strainlist = {key: np.zeros((ndata, 1, tlen), dtype=np.float32) for key in ifolist.keys()}

    for idx in range(ndata):
        if_not_exist_makedir(f'{outdir}/{idx:d}/')

        # Simulate noise
        simulated_noise = pycbc.noise.noise_from_psd(tlen, delta_t, psd_analytic)

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
        distance = rejection_sampling(pdf_dl, dlmin, dlmax) * 1000.0
        z = z_at_value(cosmo.luminosity_distance, distance * Mpc)
        params = {
            'approximant': approximant,
            'mass1': m1 * (1.0 + z),
            'mass2': m2 * (1.0 + z),
            'spin1z': a1,
            'spin2z': a2,
            'delta_t': 1.0 / fs,
            'f_lower': f_lower,
            'inclination': inclination,
            'distance': distance,
            'phi0': phi0
        }
        locparams = {
            'declination': dec,
            'right_ascension': ra,
            'polarization': pol,
            't_gps': tgps
        }
        # parameterlist[idx] = params | locparams

        # Generate waveforms
        hp, hc = get_td_waveform(**params)
        # # Prevent an artificial peak to appear in whitened strain
        # w = tukey(len(hp), alpha=1.0 / 8.0)
        # w[len(hp) // 2:] = 1.0
        # hpj, hcj = adjust_waveform_length(hp * w, hc * w, duration, merger=tcoalescence / duration)
        hpj, hcj = adjust_waveform_length(hp, hc, total_duration, merger=tcoalescence / total_duration)
        # Adjust GPS time
        hpj.start_time = tgps - tcoalescence
        hcj.start_time = tgps - tcoalescence
        simulated_noise.start_time = hpj.start_time

        # Project onto the detector frame
        fp, fc = ifo.antenna_pattern(**locparams)
        td_from_earth_center = ifo.time_delay_from_earth_center(ra, dec, tgps)
        strain = fp * hpj + fc * hcj
        strain.roll(int(td_from_earth_center / hpj.delta_t))
        strain = strain + simulated_noise
        # strainlist[key][idx, 0] = strain_wh

        # Estimate PSDs
        strain_highpass = highpass(strain, 15.0)
        psd = strain_highpass.psd(4.0)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=15)

        # Calculate SNR
        snr = matched_filter(fp * hpj + fc * hcj, strain, psd=psd, low_frequency_cutoff=20.0)
        snr = (abs(snr.crop(4, 4))).max()

        # Save data
        strain.save(f'{outdir}/{idx:d}/strain.npy')
        psd.save(f'{outdir}/{idx:d}/psd.npy')
        with open(f'{outdir}/{idx:d}/parameter.pkl', 'wb') as f:
            pickle.dump(params | locparams | {'snr': snr}, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process the noise strain')
    parser.add_argument('--outdir', type=str, help='Directory name including `train` or `validate` or `test`.')
    parser.add_argument('--ndata', type=int, help='Data number')
    args = parser.parse_args()
    main(args)
