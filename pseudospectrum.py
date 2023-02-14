import pdb
import os
import shutil
import logging
import pickle
import json
import numpy as np
from astropy.io import fits
from configparser import ConfigParser
#from lmfit import Parameters, minimize, fit_report
from scipy.io import readsav
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

from maps import Maps
from catalogs import Catalogs
from toolbox import Toolbox

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
conv_luv_to_sfr = 2.17e-10
a_nu_flux_to_mass = 6.7e19 # erg / s / Hz / Msun
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K
dtor = 0.017453292
sky_sqr_deg = 41253

class PseudoSpectrum(Maps, Catalogs, Toolbox):


    def __init__(self, param_path_file):
        super().__init__()

        # Import parameters from config.ini file
        self.config_dict = self.get_params_dict(param_path_file)

    @classmethod
    def get_k_from_map(cls, mapin, pixsize_arcsec):
        dims = np.shape(mapin)
        return cls.get_kmap(dims, pixsize_arcsec)

    def get_kmap(self, dims, pixsize_arcsec):

        lx = np.fft.fftfreq(dims[0]) * 2
        ly = np.fft.fftfreq(dims[1]) * 2
        lx = np.fft.ifftshift(lx) * (180 * 3600. / pixsize_arcsec)
        ly = np.fft.ifftshift(ly) * (180 * 3600. / pixsize_arcsec)
        ly, lx = np.meshgrid(ly, lx)
        l2d = np.sqrt(lx ** 2 + ly ** 2)
        return l2d

    def get_kmap_idl(self, dims, pixsize_arcsec):

        kx = np.arange(dims[0] // 2 + 1) / ((pixsize_arcsec * dims[0] * np.pi / 10800. / 60.)) * 2. * np.pi
        kx = np.concatenate((kx, -kx[1:((1 + dims[0]) // 2)][::-1]))

        ky = np.arange(dims[1] // 2 + 1) / ((pixsize_arcsec * dims[1] * np.pi / 10800. / 60.)) * 2. * np.pi
        ky = np.concatenate((ky, -ky[1:((1 + dims[1]) // 2)][::-1]))

        lx = len(kx)
        ly = len(ky)
        kx = (np.ones((ly, lx)) * kx)
        ky = (np.ones((lx, ly)) * ky).T
        k = np.sqrt(kx ** 2 + ky ** 2)
        return k.T

    def get_twod_fft_idl(self, map_one, map_two=None, pix_arcsec=False):
        dims = np.shape(map_one)
        if pix_arcsec:
            fac = (pix_arcsec / 3600 * (np.pi / 180)) ** 2 * (dims[0] * dims[1])
        else:
            fac = 1
        if map_two is None:
            return np.real(np.fft.fft2(map_one) * np.conj(np.fft.fft2(map_one))) * fac
        else:
            return np.real(np.fft.fft2(map_one) * np.conj(np.fft.fft2(map_two))) * fac

    def get_twod_fft(self, map_one, map_two=None, pix_arcsec=False):
        if map_two is None:
            map_two = map_one.copy()

        if pix_arcsec:
            dimx, dimy = map_one.shape
            sterad_per_pix = (pix_arcsec / 3600 / 180 * np.pi) ** 2
            V = dimx * dimy * sterad_per_pix

            ffta = np.fft.fftn(map_one * sterad_per_pix)
            fftb = np.fft.fftn(map_two * sterad_per_pix)
            ps2d = np.real(ffta * np.conj(fftb)) / V
            ps2d = np.fft.ifftshift(ps2d)
        else:
            ffta = np.fft.fftn(map_one)
            fftb = np.fft.fftn(map_two)
            ps2d = np.real(ffta * np.conj(fftb))
            ps2d = np.fft.ifftshift(ps2d)

        return ps2d

    def get_twod_ifft(self, map_one, map_two=None, pix_arcsec=False):
        dims = np.shape(map_one)
        if pix_arcsec:
            fac = (pix_arcsec / 3600 * (np.pi / 180)) ** 2 * (dims[0] * dims[1])
        else:
            fac = 1
        if map_two is None:
            return np.real(np.fft.ifft2(map_one) * np.conj(np.fft.ifft2(map_one))) * fac
        else:
            return np.real(np.fft.ifft2(map_one) * np.conj(np.fft.ifft2(map_two))) * fac

    def get_twod_ifft_new(self, map_one, map_two=None, pix_arcsec=False):
        if map_two is None:
            map_two = map_one.copy()
        dimx, dimy = map_one.shape
        if pix_arcsec:
            sterad_per_pix = (pix_arcsec / 3600 / 180 * np.pi) ** 2
            V = dimx * dimy * sterad_per_pix
        else:
            sterad_per_pix = 1
            V = 1

        ffta = np.fft.fftn(map_one * sterad_per_pix)
        fftb = np.fft.fftn(map_two * sterad_per_pix)
        ps2d = np.real(ffta * np.conj(fftb)) / V
        ps2d = np.fft.ifftshift(ps2d)
        return ps2d

    @staticmethod
    def get_map_native_deltal(mapin, pixsize_arcsec):
        dims = np.shape(mapin)
        eff_side = np.sqrt(np.product(dims))
        eff_a = eff_side * (pixsize_arcsec / 3600.) * dtor
        deltal = 2. * np.pi / eff_a
        return deltal

    def get_ell_bins(self, mapin, pix_arcsec, deltal=None, width=0, lmin=0, k_theta=False):
        if not deltal:
            shape_mapin = np.shape(mapin)
            xx = shape_mapin[0]
            yy = shape_mapin[1]
            eff_side = np.sqrt(xx * yy)
            #eff_side = yy
            print('sqrt(xx*yy) = {}, xx={}, yy={}'.format(np.sqrt(xx * yy), xx, yy))
            eff_a = eff_side * (pix_arcsec / 3600.) * dtor
            deltal = 2. * np.pi / eff_a
            print('deltal is {}'.format(deltal))

        if not width:
            dims = np.shape(mapin)
            kmap = self.get_kmap(dims, pix_arcsec)
            nk = int(np.floor(np.max(kmap) / deltal))
            ell = np.arange(nk) * deltal + deltal / 2 + lmin
        else:
            ell = self.linloggen(deltal=deltal, width=width)

        k_theta_log = self.ell_to_k(ell)
        if k_theta:
            return k_theta_log
        else:
            return ell

    @classmethod
    def bin_in_rings(cls, mapin, ell_bins, kmap, kpower=0):
        pk = np.zeros_like(ell_bins[:-1])
        deltal = np.diff(ell_bins)
        ell = ell_bins[:-1] + deltal / 2.
        ind_log = deltal > deltal[0]
        ell[ind_log] = np.sqrt(ell_bins[:-1] * ell_bins[1:])[ind_log]
        for i in range(len(ell_bins[:-1])):
            ind_ell = (kmap >= ell_bins[i]) & (kmap < ell_bins[i + 1])
            pk[i] = np.mean(mapin[ind_ell] * kmap[ind_ell] ** kpower) / \
                    ell[i] ** kpower
            #ell[i] = np.mean(kmap[ind_ell])
        return pk, ell

    def get_mkk_ell_range(self, ell_bins, kmap, mask_one, mask_two=None):
        shape_map = np.shape(mask_one)
        npk = len(ell_bins[:-1])
        pk = np.zeros([npk, npk])
        #ell = np.zeros_like(pk)
        for iell in range(npk):
            #print('Getting ell={0:0.0f} to {1:0.0f}'.format(ell_bins[iell],ell_bins[iell+1]))
            idx_ring = (kmap >= ell_bins[iell]) & (kmap < ell_bins[iell + 1])
            idx_not_ring = (kmap < ell_bins[iell]) | (kmap >= ell_bins[iell + 1])
            imap_ring = np.ones_like(mask_one) * np.random.normal(size=shape_map)
            imap_ring[idx_not_ring] = 0

            imode_map = (np.real(np.fft.ifft2(imap_ring)) + np.imag(np.fft.ifft2(imap_ring)))
            if mask_two is None:
                imask_mkk = self.get_twod_fft(imode_map, map_two=None, pix_arcsec=None)
            else:
                imask_mkk = self.get_twod_fft(imode_map * mask_one, map_two=imode_map * mask_two, pix_arcsec=None)

            #pdb.set_trace()
            ipk_mask, ipk_ell = self.bin_in_rings(imask_mkk, ell_bins, kmap)
            pk[iell] = ipk_mask
            #ell[iell] = np.mean(kmap[idx_ring])

        return pk
        #return ell, pk

    def get_mkk(self, ell_bins, kmap, mask_one, mask_two=None, iterations=1):

        for i in np.arange(iterations):
            print("Iteration {0}/{1}".format(i+1, iterations))
            if not i:
                ipk = self.get_mkk_ell_range(ell_bins, kmap, mask_one, mask_two=mask_two)
                #pdb.set_trace()
            else:
                ipk2 = self.get_mkk_ell_range(ell_bins, kmap, mask_one, mask_two=mask_two)
                ipk += ipk2

        return ipk/iterations

    def get_mkks(self,
                 maps_dict=None,
                 maps_dict_two=None,
                 cross_spectra=False,
                 iterations=1,
                 overwrite=False,
                 deltal=100,
                 width=0.35):

        mkk_dict = {}

        if maps_dict is None:
            maps_dict = self.maps_dict

        if maps_dict_two is None:
            maps_dict_two = self.maps_dict

        maps_dict_keys_one = maps_dict.keys()
        maps_dict_keys_two = maps_dict_two.keys()

        for ione, imap_one in enumerate(maps_dict_keys_one):
            for itwo, imap_two in enumerate(maps_dict_keys_two):
                fwhm_match = (maps_dict[imap_one]['fwhm'] == maps_dict_two[imap_two]['fwhm'])
                if fwhm_match or (cross_spectra is True):
                    mkk_key = 'x'.join([imap_one, imap_two, str(iterations)])
                    ell_key = '__deltal_{0:0.0f}__width_{1:0.2f}'.format(deltal, width).replace('.', 'p')
                    path_mkk_dir = self.parse_path(os.path.join(self.config_dict['io']['output_folder'])+' mkk')
                    mask_key=""
                    if 'masks' in maps_dict[imap_one]:
                        if 'hanning' in maps_dict[imap_one]['masks']:
                            mask_key = "_hanning"
                        elif 'kaiser' in maps_dict[imap_one]['masks']:
                            mask_key = "_kaiser"
                        elif 'blackman' in maps_dict[imap_one]['masks']:
                            mask_key = "_blackman"
                    mkk_filename = mkk_key+ell_key+mask_key+'.pkl'
                    #mkk_filename = mkk_key+ell_key+mask_key+'.fits'
                    path_mkk_file = os.path.join(path_mkk_dir, mkk_filename)
                    # Import if exists.  Calculate if not.
                    if os.path.isfile(path_mkk_file) and not overwrite:
                        mkk_array = Toolbox.import_saved_pickles(path_mkk_file)
                    else:
                        print('Calculating {} mkk'.format(mkk_key))
                        kmap = maps_dict[imap_one]['kmap']
                        pix_arcsec = maps_dict[imap_one]['pixel_size']
                        mask_one = maps_dict[imap_one]['masks']['mask']
                        mask_two = maps_dict_two[imap_two]['masks']['mask']
                        if 'hanning' in maps_dict[imap_one]['masks']:
                            mask_one *= maps_dict[imap_one]['masks']['hanning']
                            mask_two *= maps_dict_two[imap_two]['masks']['hanning']
                        elif 'kaiser' in maps_dict[imap_one]['masks']:
                            mask_one *= maps_dict[imap_one]['masks']['kaiser']
                            mask_two *= maps_dict_two[imap_two]['masks']['kaiser']
                        elif 'blackman' in self.maps_dict[imap_one]['masks']:
                            mask_one *= maps_dict[imap_one]['masks']['blackman']
                            mask_two *= maps_dict_two[imap_two]['masks']['blackman']
                        ell_bins = self.get_ell_bins(mask_one, pix_arcsec, deltal=deltal, width=width)
                        mkk_array = self.get_mkk(ell_bins, kmap, mask_one, mask_two=mask_two, iterations=iterations)
                        Toolbox.save_to_pickles(path_mkk_file, mkk_array)

                    mkk_dict[mkk_key] = mkk_array
        return mkk_dict

    #def get_cross_pseudospectrum(self, map_object1, map_object2=None):

    #def get_pseudospectrum(self, map_object1, map_object2=None):

    def get_pseudospectra(self,
                          mkk_dict=None,
                          maps_dict=None,
                          maps_dict_two=None,
                          deltal=100, width=0.35,
                          iterations=20,
                          overwrite=False,
                          cross_spectra=False
                          ):
        # Loop over maps
        pk_dict_out = {}

        if maps_dict is None:
            maps_dict = self.maps_dict

        if maps_dict_two is None:
            maps_dict_two = self.maps_dict

        for ione, imap_one in enumerate(maps_dict.keys()):
            for itwo, imap_two in enumerate(maps_dict_two.keys()):
                # compare beamsizes cross_spectra==True is when beam sizes do not match
                fwhm_match = (maps_dict[imap_one]['fwhm'] == maps_dict_two[imap_two]['fwhm'])
                if fwhm_match or (cross_spectra is True):
                    mask_key = ""
                    if 'masks' in maps_dict[imap_one]:
                        if 'hanning' in maps_dict[imap_one]['masks']:
                            mask_key = '_hanning'
                        if 'blackman' in maps_dict[imap_one]['masks']:
                            mask_key = 'blackman'
                        if 'kaiser' in maps_dict[imap_one]['masks']:
                            mask_key = 'kaiser'
                    pk_key = 'X'.join([imap_one, imap_two, str(iterations)])
                    ell_key = '__deltal_{0:0.0f}__width_{1:0.2f}'.format(deltal, width).replace('.', 'p')
                    pk_filename = pk_key + ell_key + '.pkl'
                    path_pk_dir = self.parse_path(os.path.join(self.config_dict['io']['output_folder']) + ' pk')
                    #path_pk_file = os.path.join(path_pk_dir, pk_filename)
                    #print(path_pk_file)
                    path_pk_file = os.path.join(self.config_dict['io']['saved_data_path'], pk_filename)
                    fft_key = 'X'.join([imap_one, imap_two])
                    fft_filename = fft_key + mask_key + '.fits' #'.pkl'
                    path_fft_dir = self.parse_path(os.path.join(self.config_dict['io']['output_folder']) + ' fft')
                    path_fft_file = os.path.join(path_fft_dir, fft_filename)

                    # Import or calculate FFT
                    if os.path.isfile(path_fft_file) and not overwrite:
                        #fft_array = Toolbox.import_saved_pickles(path_fft_file)
                        fft_array = fits.getdata(path_fft_file, 0, header=False)
                    else:
                        print('Calculating {} pk'.format(pk_key))
                        kmap = maps_dict[imap_one]['kmap']
                        pix_arcsec = maps_dict[imap_one]['pixel_size']
                        wavelength_one = maps_dict[imap_one]['wavelength']
                        wavelength_two = maps_dict_two[imap_two]['wavelength']

                        # get maps
                        map_one = maps_dict[imap_one]['map']
                        map_two = maps_dict_two[imap_two]['map']
                        ell_bins = self.get_ell_bins(map_one, pix_arcsec, deltal=deltal, width=width)

                        # get masks
                        mask_one = maps_dict[imap_one]['masks']['mask']
                        mask_two = maps_dict_two[imap_two]['masks']['mask']
                        if 'hanning' in maps_dict[imap_one]['masks']:
                            mask_one *= maps_dict[imap_one]['masks']['hanning']
                            mask_two *= maps_dict_two[imap_two]['masks']['hanning']
                        elif 'kaiser' in maps_dict[imap_one]['masks']:
                            mask_one *= maps_dict[imap_one]['masks']['kaiser']
                            mask_two *= maps_dict_two[imap_two]['masks']['kaiser']
                        elif 'blackman' in maps_dict[imap_one]['masks']:
                            mask_one *= maps_dict[imap_one]['masks']['blackman']
                            mask_two *= maps_dict_two[imap_two]['masks']['blackman']

                        #fft_array = self.get_twod_ifft(map_one * mask_one, map_two * mask_two, pix_arcsec=pix_arcsec)
                        fft_array = self.get_twod_fft(map_one * mask_one, map_two * mask_two, pix_arcsec=pix_arcsec)

                        hdu = fits.PrimaryHDU(fft_array, header=self.maps_dict[imap_one]['header'])
                        hdul = fits.HDUList([hdu])
                        hdul.writeto(path_fft_file, overwrite=True)

                    # Import or Calculate PK
                    if os.path.isfile(path_pk_file) and not overwrite:
                        pk_dict = Toolbox.import_saved_pickles(path_pk_file)
                        #Toolbox.save_to_pickles(path_fft_file, fft_array)
                    else:
                        print('Calculating {} pk'.format(pk_key))
                        kmap = maps_dict[imap_one]['kmap']
                        pix_arcsec = maps_dict[imap_one]['pixel_size']
                        wavelength_one = maps_dict[imap_one]['wavelength']
                        wavelength_two = maps_dict_two[imap_two]['wavelength']

                        ell_bins = self.get_ell_bins(fft_array, pix_arcsec, deltal=deltal, width=width)
                        pk_array, ell = self.bin_in_rings(fft_array, ell_bins, kmap)

                        '''
                        ell = np.sqrt(ell_bins[:-1] * ell_bins[1:])
                        nbins=len(ell)
                        ell, pk_array = self.Cl_from_map2D(map_a=map_one, map_b=map_two, mask=mask_one,
                                                           pixsize=pix_arcsec, lbinedges=ell_bins, lbins=ell,
                                                           nbins=nbins, logbin=False, weights=None, return_full=False,
                                                           return_Dl=False, interp_Cl=False)
                        '''

                        # get beam
                        try:
                            map_fwhm_one = maps_dict[imap_one]['beam']['fwhm']
                            map_fwhm_two = maps_dict_two[imap_two]['beam']['fwhm']
                        except:
                            map_fwhm_one = maps_dict[imap_one]['fwhm']
                            map_fwhm_two = maps_dict_two[imap_two]['fwhm']
                        Bl1 = self.get_psf_correction(ell, map_fwhm_one, map_fwhm_one)
                        Bl2 = self.get_psf_correction(ell, map_fwhm_two, map_fwhm_two)
                        Bl = self.get_psf_correction(ell, map_fwhm_one, map_fwhm_two)
                        ipix = self.pixel_beam_function(ell, pix_arcsec)

                        # save pk
                        pk_dict = {'ell': ell,
                                   'psf1': Bl1,
                                   'psf2': Bl2,
                                   'Bl': Bl,
                                   'pix': ipix,
                                   'epsf': ipix*Bl,
                                   'wv1': wavelength_one,
                                   'wv2': wavelength_two,
                                   'pk_raw': pk_array,
                                   'pk_beam_corrected': pk_array / Bl / ipix
                                   }

                        # Correct MKK if given.
                        if mkk_dict is not None:
                            mkk = np.linalg.inv(mkk_dict[pk_key.lower()])
                            pk_dict['pk_mkk_corrected'] = np.matmul(pk_array, mkk)
                            pk_dict['pk_mkk_and_beam_corrected'] = np.matmul(pk_array, mkk) / Bl

                        Toolbox.save_to_pickles(path_pk_file, pk_dict)

                    pk_dict_out[fft_key.lower()] = pk_dict

        return pk_dict_out
