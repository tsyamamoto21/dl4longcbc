import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
import pycbc.noise
import pycbc.psd
from pycbc.filter import resample_to_delta_t, highpass, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from tqdm.notebook import tqdm


def main(args):
    # Random seed
    seed = args.seed
    # datadir
    if args.datadir is None:
        datadir = './data/240926mfpattern/'
    else:
        datadir = args.datadir

    # Strain parameters
    fs = 4096
    duration = 32
    delta_t = 1.0 / fs
    flow = 5.0
    delta_f = 1.0 / duration
    flen = int(fs / 2 / delta_f) + 1

    # PSD analytic
    psd_analytic = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    # Generate {duration} seconds of noise at {fs} Hz
    tsamples = int(duration / delta_t)
    strain = pycbc.noise.noise_from_psd(tsamples, delta_t, psd_analytic, seed=seed)
    # Remove the low frequency content and downsample the data to 2048Hz
    strain = highpass(strain, 15.0)

    # PSD
    psd = strain.psd(4)
    psd = pycbc.psd.interpolate(psd, strain.delta_f)
    psd = pycbc.psd.inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=15)

    # Inject signal
    snrinj = 30
    # mc_inj = 25.0
    # q_inj = 0.125
    # m1_inj = mass1_from_mchirp_q(mc_inj, q_inj)
    # m2_inj = mass2_from_mchirp_q(mc_inj, q_inj)
    m1_inj = 85.0
    m2_inj = 66.0
    hpinj, _ = get_td_waveform(
        approximant='IMRPhenomD',
        mass1=m1_inj,
        mass2=m2_inj,
        delta_t=strain.delta_t,
        f_lower=10.0,
    )
    hpinj.resize(len(strain))
    t_before_merger = - hpinj.sample_times[0]
    hpinj.roll(int((duration // 2 - t_before_merger) * fs))
    hpinj.start_time = strain.start_time

    # Template
    template, _ = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1_inj,
        mass2=m2_inj,
        delta_f=strain.delta_f,
        f_lower=20.0
    )
    # Resize the vector to match our data
    template.resize(len(psd))
    # Calculate SNR
    snr = matched_filter(template, strain + hpinj, psd=psd, low_frequency_cutoff=20)
    snr_cropped = snr.crop(4, 4)
    snr_tentative = (abs(snr_cropped)).max()
    strain_injected = strain + hpinj / snr_tentative * snrinj
    snr = matched_filter(template, strain_injected, psd=psd, low_frequency_cutoff=20)
    snr_cropped = snr.crop(4, 4)

    # Generate images
    # Various mass ratio
    q_temp_list = [1.0, 0.5, 0.25, 0.125]
    spin1z = 0.0
    spin2z = 0.0

    # strainlen = len(strain_injected)
    approximant = 'IMRPhenomD'
    mcmin = 5.0
    mcmax = 45.0
    flg_logmc = False
    if flg_logmc:
        Mclist = np.logspace(np.log10(mcmin), np.log10(mcmax), 75)
    else:
        Mclist = np.linspace(mcmin, mcmax, 75, endpoint=True)  # Detector frame
    ntemp = len(Mclist)
    cropi = 4 * 4096
    crope = 4 * 4096
    sampletime = strain_injected.sample_times[cropi: -crope]

    tleft = duration / 2 - 0.15
    tright = duration / 2 + 0.15
    kcrop_left = np.argmin(abs(sampletime - tleft))
    kcrop_right = np.argmin(abs(sampletime - tright))
    # mupper = Mclist.max()
    # mlower = Mclist.min()
    sampletime_cropped = sampletime[kcrop_left: kcrop_right]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True, layout='constrained')
    for i, q_temp in enumerate(q_temp_list):
        nx = i // 2
        ny = i % 2
        snrlist = np.zeros((ntemp, kcrop_right - kcrop_left))

        for idx, mc in enumerate(tqdm(Mclist)):

            mass1 = mass1_from_mchirp_q(mc, q_temp)
            mass2 = mass2_from_mchirp_q(mc, q_temp)

            # Generate a template to filter with
            params = {
                'approximant': approximant,
                'mass1': mass1,
                'mass2': mass2,
                'spin1z': spin1z,
                'spin2z': spin2z,
                'f_lower': 20.0,
                'delta_f': 1.0 / strain.duration
            }
            template, _ = get_fd_waveform(**params)
            template.resize(len(psd))
            # Calculate the complex (two-phase SNR)
            snr = matched_filter(template, strain_injected, psd=psd, low_frequency_cutoff=20)
            snrlist[idx] = abs(snr.crop(4, 4))[kcrop_left: kcrop_right]

        cb = ax[nx, ny].pcolormesh(sampletime_cropped, Mclist, snrlist, vmin=0, vmax=30, cmap='inferno')
        # ax[nx, ny].axhline(mc_inj, c='w', linestyle=':')
        ax[nx, ny].set_title(f'q={q_temp}')
        if flg_logmc:
            ax[nx, ny].set_yscale('log')
    ax[0, 0].set(ylabel='Chirp mass [M_sun]')
    ax[1, 0].set(xlabel='Time [sec]', ylabel='Chirp mass [M_sun]')
    ax[1, 1].set(xlabel='Time [sec]')

    # # Colorbar
    # axpos = ax[1, 1].get_position()
    # cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
    # norm = colors.Normalize(vmin=0, vmax=30)
    # mappable = ScalarMappable(cmap='inferno', norm=norm)
    # mappable._A = []
    # fig.colorbar(mappable, cax=cbar_ax)
    # cbar = fig.colorbar(cb, ax=ax.ravel().tolist(), pad=-0.25)
    fig.colorbar(cb, ax=ax, aspect=25)
    if flg_logmc:
        fig.savefig(f'{datadir}/mfpattern_logmc_{mc_inj:.2f}_{1.0/q_inj:.1f}.png');
    else:
        fig.savefig(f'{datadir}/mfpattern_{mc_inj:.2f}_{1.0/q_inj:.1f}.png');



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Make matched filter highmass BBH signals injected into Gaussian noise.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--datadir', type=str, default=None, help='Directory path where the data will be saved.')
    args = parser.parse_args()
    main(args)
