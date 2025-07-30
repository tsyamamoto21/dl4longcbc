#!/usr/bin/env python
import pickle
import numpy as np
import torch
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy.units import Mpc
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta, mchirp_from_mass1_mass2
from pycbc.filter import highpass, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.detector import Detector
import pycbc.noise
from dl4longcbc.sampler import rejection_sampling
from dl4longcbc.utils import adjust_waveform_length, if_not_exist_makedir


def make_snrmap_coarse(snrmap, kfilter):
    nc, nx, ny = snrmap.shape
    ny_coarse = ny // kfilter
    snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
    for i in range(ny_coarse):
        snrmap_coarse[:, :, i] = torch.sqrt(torch.mean(snrmap[:, :, i * kfilter: (i + 1) * kfilter]**2, dim=-1))
    return snrmap_coarse


class SignalProcessingParameters:
    def __init__(self, duration, tsegment, fs, low_frequency_cutoff, tfft, tukey_alpha, width_input, height_input):
        # Signal properties
        self.duration = duration
        self.tsegment = tsegment
        self.fs = fs
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = self.fs / 2

        # FFT for PSD estimation
        self.tfft = tfft
        self.toverlap = self.tfft / 2
        self.fftlength = int(self.tfft * self.fs)
        self.overlaplength = int(self.fftlength / 2)

        # MF params
        self.mfdatalength = int(self.duration * self.fs)
        self.tukey_alpha = tukey_alpha

        # Image properties
        self.width_input = width_input
        self.height_input = height_input
        self.kfilter = int(self.fs * self.tsegment / self.width_input)
        self.kcrop_left = int(self.fs * (self.duration / 2 - 3 * self.tsegment / 4))
        self.kcrop_right = int(self.fs * (self.duration / 2 + 3 * self.tsegment / 4))
        self.width_before_smearing = self.kcrop_right - self.kcrop_left


def generatestrain_matchedfilter_core(file_parameter: str, template_bank: list, outdir: str, sp: SignalProcessingParameters, noiseonly=False):

    print(f"[PID {os.getpid()}] Processing {timestamps.start_time}: Binded to {os.sched_getaffinity(0)}")
    # SNR array
    snrlist = torch.zeros((2, sp.height_input, sp.width_before_smearing), requires_grad=False)

    # Tukey window
    window = tukey(sp.mfdatalength, sp.tukey_alpha)

    tclist_for_short_segment = np.array(timestamps.tclist_for_short_segment)
    if offevent:
        tclist_for_short_segment = (tclist_for_short_segment[:-1] + tclist_for_short_segment[1:]) / 2

    # Load a hdf file.
    xh1 = load_timeseries(file_foreground, group=f'H1/{timestamps.start_time_str}')
    xl1 = load_timeseries(file_foreground, group=f'L1/{timestamps.start_time_str}')

    # Estimate PSD
    psd_h = pycbc.psd.welch(xh1, seg_len=sp.fftlength, seg_stride=sp.overlaplength, avg_method='median-mean')
    psd_h_interp = pycbc.psd.interpolate(psd_h, delta_f=1.0 / sp.duration)
    psd_l = pycbc.psd.welch(xl1, seg_len=sp.fftlength, seg_stride=sp.overlaplength, avg_method='median-mean')
    psd_l_interp = pycbc.psd.interpolate(psd_l, delta_f=1.0 / sp.duration)
    print(f"[PID {os.getpid()}] Processing {timestamps.start_time}: PSD estimated")

    dataidx = 0
    for tc in tclist_for_short_segment:
        tini = tc - sp.duration / 2
        tfin = tc + sp.duration / 2
        if (timestamps.start_time < tini) and (tfin < timestamps.end_time):
            # Slice the data and window
            strain_h = xh1.time_slice(tini, tfin) * window
            strain_l = xl1.time_slice(tini, tfin) * window

            # Calculate SNR
            for i in range(sp.height_input):
                rho_h = matched_filter(template_bank[i], strain_h, psd=psd_h_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
                rho_l = matched_filter(template_bank[i], strain_l, psd=psd_l_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
                snrlist[0, i] = torch.from_numpy(abs(rho_h).numpy())[sp.kcrop_left: sp.kcrop_right]
                snrlist[1, i] = torch.from_numpy(abs(rho_l).numpy())[sp.kcrop_left: sp.kcrop_right]

            dataavg = make_snrmap_coarse(snrlist, sp.kfilter).to(torch.float32)
            torchfilename = f'{outdir}/inputs_{timestamps.start_time_str}_{int(sp.duration):d}_{dataidx:d}.pth'
            torch.save(dataavg, torchfilename)
            # print(f'[PID {os.getpid()}] Torch file ({torchfilename}) is saved.')
            dataidx += 1


def main(args):
    outdir = args.outdir
    if_not_exist_makedir(outdir)
    ndata = args.ndata

    # Strain parameters
    ifonamelist = ['H1', 'L1']
    ifodict = {ifoname: Detector(ifoname) for ifoname in ifonamelist}
    fs = 2048
    dt = 1.0 / fs
    duration = 32
    tcoalescence = duration / 2
    tlen = int(fs * duration)
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

    # Image parameters
    image_size_time = 0.75
    t_snrcrop = 4.0
    kcenter = int(fs * (duration - 2 * t_snrcrop) / 2)
    kleft = kcenter - int(image_size_time * fs / 2)
    kright = kcenter + int(image_size_time * fs / 2)
    tfilter = 16

    sp = SignalProcessingParameters(
        duration=16,
        tsegment=1,
        fs=2048,
        low_frequency_cutoff=20.0,
        tukey_alpha=1.0 / 16.0,
        tfft=4.0,
        width_input=256,
        height_input=256
    )

    # Waveform parameters
    approximant = 'IMRPhenomD'
    mcmin_src, mcmax_src = 50.0, 200.0
    dlmin, dlmax = 0.3, 2.0  # Gpc
    # MDC-1 setting is employed here.
    # dlcmin, dlcmax = 0.130, 0.350  # Gpc
    # m1min, m1max = 10.0, 50.0
    # mc0 = mchirp_from_mass1_mass2(1.4, 1.4)

    def pdf_dl(x):
        return 3.0 * x**2 / (dlmax**3 - dlmin**3)

    # Make a template bank
    approximant_tmp = 'IMRPhenomXPHM'
    mcmin_tmp = 5.0
    mcmax_tmp = 100.0
    ngrid_mc = sp.height_input
    mclist = np.logspace(np.log10(mcmin_tmp), np.log10(mcmax_tmp), ngrid_mc, endpoint=True)
    eta = 0.25
    a1 = 0.0
    a2 = 0.0
    template_bank = []
    for i in range(ngrid_mc):
        mass1 = mass1_from_mchirp_eta(mclist[i], eta)
        mass2 = mass2_from_mchirp_eta(mclist[i], eta)
        params_tmp = {
            'approximant': approximant_tmp,
            'mass1': mass1,
            'mass2': mass2,
            'spin1z': a1,
            'spin2z': a2,
            'f_lower': sp.low_frequency_cutoff,
            'delta_f': 1.0 / sp.duration,
            'f_final': sp.high_frequency_cutoff
        }

        hp_fd, _ = get_fd_waveform(**params_tmp)
        template_bank.append(hp_fd)

    # Generate noise

    # Inject signal

    # Carry out matched filter


    for idx_data in range(ndata):
        if args.no_injection:
            subdirectory = f'{outdir}/noise/'
        else:
            subdirectory = f'{outdir}/cbc/'
        if_not_exist_makedir(subdirectory)

        # Simulate noise
        simulated_noise = pycbc.noise.noise_from_psd(tlen, dt, psd_analytic)

        if args.no_injection:
            strain = simulated_noise

        else:
            # Simulate signal
            mc = np.random.uniform(mcmin_src, mcmax_src)
            eta = 0.25
            m1 = mass1_from_mchirp_eta(mc, eta)
            m2 = mass2_from_mchirp_eta(mc, eta)
            # m1 = np.random.uniform(m1min, m1max)
            # m2 = np.random.uniform(m1min, m1max)
            if m1 < m2:
                m1, m2 = m2, m1
            mc = mchirp_from_mass1_mass2(m1, m2)
            a1 = 0.0
            a2 = 0.0
            dec = np.arcsin(np.random.uniform(-1, 1))
            ra = np.random.uniform(0.0, 2.0 * np.pi)
            pol = np.random.uniform(0.0, np.pi)
            inclination = np.arcsin(np.random.uniform(-1.0, 1.0))
            phi0 = np.random.uniform(0.0, 2.0 * np.pi)
            distance = rejection_sampling(pdf_dl, dlmin, dlmax) * 1000.0
            # dlc = rejection_sampling(pdf_dl, dlcmin, dlcmax) * 1000.0
            # distance = dlc * ((mc / mc0)**(5.0 / 6.0))
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

            # Genrerate waveforms
            hp, hc = get_td_waveform(**params)
            hpj, hcj = adjust_waveform_length(hp, hc, duration, merger=tcoalescence / duration)

            # Adjust GPS time
            hpj.start_time = tgps - tcoalescence
            hcj.start_time = tgps - tcoalescence
            simulated_noise.start_time = hpj.start_time

            # Project onto the detector frame
            fp, fc = ifo.antenna_pattern(**locparams)
            td_from_earth_center = ifo.time_delay_from_earth_center(ra, dec, tgps)
            signal_inject = fp * hpj + fc * hcj
            signal_inject.roll(int(td_from_earth_center / hpj.delta_t))
            strain = signal_inject + simulated_noise

        # Estimate PSDs
        strain_highpass = highpass(strain, 15.0)
        psd = strain_highpass.psd(4.0)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=15)

        if not args.no_injection:
            # Calculate SNR
            signal_inject.roll(int(tcoalescence * fs))
            snr = matched_filter(signal_inject, strain, psd=psd, low_frequency_cutoff=20.0)
            # snr.save(f'{subdirectory}/snr_{idx_data:d}.npy')
            snr = (abs(snr.crop(4, 4))).max()
            # Save parameters
            with open(f'{subdirectory}/parameter_{idx_data:d}.pkl', 'wb') as f:
                pickle.dump(params | locparams | {'snr': snr}, f)

        # Template search
        snrlist = np.zeros((ngrid_mc, kright - kleft))
        for idx, mc in enumerate(mclist):
            mass1 = mass1_from_mchirp_eta(mc, eta)
            mass2 = mass2_from_mchirp_eta(mc, eta)

            # Generate a template to filter with
            params_tmp = {
                'approximant': approximant_tmp,
                'mass1': mass1,
                'mass2': mass2,
                'spin1z': a1,
                'spin2z': a2,
                'f_lower': 10.0,
                'delta_f': 1.0 / strain.duration
            }
            template, _ = get_fd_waveform(**params_tmp)
            template.resize(len(psd))
            # Calculate the complex (two-phase SNR)
            snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=20.0)
            snrlist[idx] = abs(snr.crop(t_snrcrop, t_snrcrop))[kleft: kright]

        snrmap_coarse = make_snrmap_coarse(snrlist, tfilter)
        s1, s2 = snrmap_coarse.shape
        snrmap_tensor = torch.tensor(snrmap_coarse, dtype=torch.float32).view(1, s1, s2)
        if args.no_injection:
            torch.save(snrmap_tensor, f'{subdirectory}/noise_{idx_data:d}.pth')
        else:
            torch.save(snrmap_tensor, f'{subdirectory}/cbc_{idx_data:d}.pth')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process the noise strain')
    parser.add_argument('--outdir', type=str, help='Directory name including `train` or `validate` or `test`.')
    parser.add_argument('--ndata', type=int, help='Data number')
    parser.add_argument('--no_injection', action='store_true', help='If true, no GW signal are injected.')
    args = parser.parse_args()
    main(args)
