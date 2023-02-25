import pdb
import os
import numpy as np
from scipy.io import readsav
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from lmfit import Parameters, minimize, fit_report
#import corner

from toolbox import Toolbox

font = {'size': 14}
plt.rc('font', **font)

path_sav = os.path.join(os.environ['MAPSPATH'],'Herschel','bandpowers','combined_spectra_all_corrected_errors')
file_sav = 'combined_spectra_extended_sources_only_dl_160_bin_width_25_percent_'

class PseudoSpectrumPlots(Toolbox):

    def __init__(self, PseudoSpectrumPlots):
        super().__init__()

        dict_list = dir(PseudoSpectrumPlots)
        for i in dict_list:
            if '__' not in i:
                setattr(self, i, getattr(PseudoSpectrumPlots, i))

    def plot_pseudospectra(self, pk_dict):
        pk_keys = pk_dict.keys()

        fig, axs = plt.subplots(len(pk_keys), len(pk_keys), figsize=(22, 10))
        for i, key in enumerate(pk_keys):
            wv_split = key.split('x')
            xcorr_label = '{0:0.0f}x{1:0.0f}'.format(pk_dict[key]['wv1'], pk_dict[key]['wv2'])
            iii = np.mod(i, len(pk_keys))

            #sav_label = 'x'.join([wv_split[0], wv_split[1]])
            path_v12 = os.path.join(path_sav, file_sav + xcorr_label + '.sav')
            ell = pk_dict[key]['ell']
            k_theta = self.ell_to_k(ell)

            #Bl = ell = pk_dict[key]['psf']

            if os.path.isfile(path_v12):
                x = readsav(path_v12)
                y = x['combined_spectra'][:, 0, 0]
                axs[i, iii].plot(x['k_t'], y, 's', c='c', label='V12')
            else:
                print('No file found {}'.format(path_v12))

            pk = pk_dict[key]['pk_raw']
            pk_corrected_psf = pk_dict[key]['pk_beam_corrected']

            plot_mkk = False
            if 'pk_mkk_corrected' in pk_dict[key]:
                pk_corrected_mkk = pk_dict[key]['pk_mkk_corrected']
                pk_corrected_both = pk_dict[key]['pk_mkk_and_beam_corrected']
                plot_mkk = True

            if (wv_split[0] == wv_split[1]) & (i == iii) | (wv_split[0] != wv_split[1]) & (i < iii):
                axs[i, iii].plot(k_theta, pk, '-o', c='g', label='spt raw')
                if plot_mkk:
                    axs[i, iii].plot(k_theta, pk_corrected_mkk, '-o', c='b', label='spt mkk corr')
                    axs[i, iii].plot(k_theta, pk_corrected_both, '-o', c='r', label='spt beam corr')

                axs[i, iii].set_xlim([0.005, 2.05])
                axs[i, iii].set_xscale('log')
                axs[i, iii].set_yscale('log')
                axs[i, iii].set_title(key)

                axs[i, iii].set_ylabel('P(k_theta) (Jy^2 sr^-1)')
                if i != iii:
                    axs[i, iii].set_xticklabels([])
                else:
                    axs[i, iii].set_xlabel('k_theta (arcmin^-1)')

                if not i and not iii:
                    axs[i, iii].legend(loc='lower left');
            else:
                axs[i, iii].axis('off')

    def plot_crossspectra(self, pk_dict):
        pk_keys = pk_dict.keys()
        x_left = np.unique([i.split('x')[0] for i in pk_dict.keys()]).tolist()[::-1]
        x_right = np.unique([i.split('x')[1] for i in pk_dict.keys()]).tolist()[::-1]

        fig, axs = plt.subplots(len(x_left), len(x_right), figsize=(22, 10))
        for i, key in enumerate(pk_keys):
            # print(key)
            wv_split = key.split('x')

            xcorr_label = '{0:0.0f}x{1:0.0f}'.format(pk_dict[key]['wv1'], pk_dict[key]['wv2'])

            ii = x_left.index(wv_split[0])
            iii = x_right.index(wv_split[1])

            # sav_label = 'x'.join([wv_split[0], wv_split[1]])
            path_v12 = os.path.join(path_sav, file_sav + xcorr_label + '.sav')
            ell = pk_dict[key]['ell']
            k_theta = self.ell_to_k(ell)

            if (wv_split[0] == wv_split[1]) & (ii == iii) | (wv_split[0] != wv_split[1]) & (ii < iii):
                # Bl = ell = pk_dict[key]['psf']

                if os.path.isfile(path_v12):
                    x = readsav(path_v12)
                    y = x['combined_spectra'][:, 0, 0]
                    axs[ii, iii].plot(x['k_t'], y, 's', c='c', label='V12')
                else:
                    print('No file found {}'.format(path_v12))

                pk = pk_dict[key]['pk_raw']
                pk_corrected_psf = pk_dict[key]['pk_beam_corrected']

                plot_mkk = False
                if 'pk_mkk_corrected' in pk_dict[key]:
                    pk_corrected_mkk = pk_dict[key]['pk_mkk_corrected']
                    pk_corrected_both = pk_dict[key]['pk_mkk_and_beam_corrected']
                    plot_mkk = True

                axs[ii, iii].plot(k_theta, pk, '-o', c='g', label='spt raw')
                if plot_mkk:
                    axs[ii, iii].plot(k_theta, pk_corrected_mkk, '-o', c='b', label='spt mkk corr')
                    axs[ii, iii].plot(k_theta, pk_corrected_both, '-o', c='r', label='spt beam corr')

                axs[ii, iii].set_xlim([0.005, 2.05])
                axs[ii, iii].set_xscale('log')
                axs[ii, iii].set_yscale('log')
                axs[ii, iii].set_title(key)

                axs[ii, iii].set_ylabel('P(k_theta) (Jy^2 sr^-1)')
                if ii != iii:
                    axs[ii, iii].set_xticklabels([])
                else:
                    axs[ii, iii].set_xlabel('k_theta (arcmin^-1)')

                if not ii and not iii:
                    axs[ii, iii].legend(loc='lower left');
            else:
                axs[ii, iii].axis('off')

        axs[1, 0].axis('off')
        axs[2, 0].axis('off')
        axs[2, 1].axis('off')