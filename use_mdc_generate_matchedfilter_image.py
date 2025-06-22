#!/usr/bin/env python
import h5py
import torch
import numpy as np
from pycbc.types import load_timeseries
import pycbc.psd
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter


def make_snrmap_coarse(snrmap, kfilter):
    nc, nx, ny = snrmap.shape
    ny_coarse = ny // kfilter
    snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
    for i in range(ny_coarse):
        snrmap_coarse[:, i] = torch.mean(snrmap[:, :, i * kfilter: (i + 1) * kfilter], dim=1)
    return snrmap_coarse

# Dataset directory
datadir = './data/mdc/ds1_2/'

# Parameters for template matching
duration = 32.0
fs = 2048
fhigh = fs / 2
low_frequency_cutoff = 20.0
fftlength = int(4 * fs)
overlap_length = int(fftlength / 2)

approximant_tmp = 'IMRPhenomXPHM'
mcmin_tmp = 5.0
mcmax_tmp = 50.0
ngrid_mc = 256
mclist = np.logspace(np.log10(mcmin_tmp), np.log10(mcmax_tmp), ngrid_mc, endpoint=True)
eta = 0.25
a1 = 0.0
a2 = 0.0

# Image parameters
height = ngrid_mc
width = 256
buffer = width // 4
width_tot = width + 2 * buffer


# Make a template bank
template_bank = []
for i in range(ngrid_mc):
    mass1 = mass1_from_mchirp_eta(mclist[i], eta)
    mass2 = mass2_from_mchirp_eta(mclist[i], eta)
    params_tmp = {
        'approximant': 'IMRPhenomD',
        'mass1': mass1,
        'mass2': mass2,
        'spin1z': 0.0,
        'spin2z': 0.0,
        'f_lower': low_frequency_cutoff,
        'delta_f': 1.0 / duration,
        'f_final': fhigh
    }

    hp_fd, _ = get_fd_waveform(**params_tmp)
    template_bank.append(hp_fd)

# Get start time and end time
with h5py.File(f'{datadir}/foreground.hdf', 'r') as fo:
    x = fo['H1'].keys()
    start_time_strlist = [xsample for xsample in x]
    start_time_list = [int(s) for s in start_time_strlist]
    nsegment = len(start_time_list)
    end_time_list = [int(len(fo['H1'][start_time_strlist[i]]) / fs) + start_time_list[i] for i in range(nsegment)]

# Get the injection time stamps
with h5py.File(f'{datadir}/injection.hdf', 'r') as fo:
    tclist_from_hdf5 = fo['tc'][:]
# Assign a segment to an injection
tclist = [[] for _ in range(nsegment)]
for idx in range(len(tclist_from_hdf5)):
    for n in range(nsegment):
        if (start_time_list[n] <= tclist_from_hdf5[idx]) * (tclist_from_hdf5[idx] <= end_time_list[n]):
            tclist[n].append(tclist_from_hdf5[idx])

# Run the main code
# snrlist = np.zeros((ngrid_mc, int(duration * fs)), dtype=np.float64)
snrlist = torch.zeros((2, ngrid_mc, int(duration * fs)), dtype=torch.float32, requires_grad=False)
dataidx = 0
for n in range(nsegment):
    # Set time stampes
    start_time_str = start_time_strlist[n]
    start_time = start_time_list[n]
    end_time = end_time_list[n]
    tclist_for_short_segment = tclist[n]

    # Load a hdf file.
    xh1 = load_timeseries(f'{datadir}/foreground.hdf', group=f'H1/{start_time_str}')
    xl1 = load_timeseries(f'{datadir}/foreground.hdf', group=f'L1/{start_time_str}')

    # Estimate PSD
    psd_h = pycbc.psd.welch(xh1, seg_len=fftlength, seg_stride=overlap_length, avg_method='median-mean')
    psd_h_interp = pycbc.psd.interpolate(psd_h, delta_f=1.0 / duration)
    psd_l = pycbc.psd.welch(xl1, seg_len=fftlength, seg_stride=overlap_length, avg_method='median-mean')
    psd_l_interp = pycbc.psd.interpolate(psd_l, delta_f=1.0 / duration)

    for tc in tclist_for_short_segment:
        tini = tc - duration / 2
        tfin = tc + duration / 2
        if (start_time < tini) and (tfin < end_time):
            print(tc)
            # Slice the data
            strain_h = xh1.time_slice(tini, tfin)
            strain_l = xl1.time_slice(tini, tfin)

            # Calculate SNR
            # rholist = []
            for i in range(ngrid_mc):
                rho_h = matched_filter(template_bank[i], strain_h, psd=psd_h_interp, low_frequency_cutoff=low_frequency_cutoff)
                rho_l = matched_filter(template_bank[i], strain_l, psd=psd_l_interp, low_frequency_cutoff=low_frequency_cutoff)
                # snrlist[i] = torch.from_numpy(((abs(rho_h)**2 + abs(rho_l)**2) ** 0.5).numpy().astype(np.float32))
                snrlist[0, i] = torch.from_numpy(abs(rho_h).numpy().astype(np.float32))
                snrlist[1, i] = torch.from_numpy(abs(rho_l).numpy().astype(np.float32))

            # save the data
            # with open(f'{datadir}/inputs_{dataidx:d}.pkl', 'wb') as fo:
            #     pickle.dump(snrlist, fo)
            # np.save(f'{datadir}/inputs_{dataidx:d}.npy', snrlist)
            torch.save(snrlist, f'{datadir}/inputs_{int(duration):d}_{dataidx:d}.pth')
            dataidx += 1
            if dataidx == 10:
                import sys
                sys.exit()
