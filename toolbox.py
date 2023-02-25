#import pdb
import os
import shutil
import logging
import pdb
import pickle
import json
import numpy as np
#from astropy.io import fits
from configparser import ConfigParser
#from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

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

class Toolbox:

    def __init__(self):
        super().__init__()

    def get_params_dict(self, param_file_path):
        config = ConfigParser()
        config.read(param_file_path)

        dict_out = {}
        for section in config.sections():
            dict_sect = {}
            for (each_key, each_val) in config.items(section):
                # Remove quotations from dicts
                try:
                    dict_sect[each_key] = json.loads(each_val)
                except:
                    dict_sect[each_key] = each_val.replace("'", '"')

            dict_out[section] = dict_sect

        # Further remove quotations from embedded dicts
        for dkey in dict_out:
            for vkey in dict_out[dkey]:
                try:
                    dict_out[dkey][vkey] = json.loads(dict_out[dkey][vkey])
                except:
                    #pdb.set_trace()
                    pass

        return dict_out

    def copy_config_file(self, fp_in, overwrite_results=False):
        ''' Place copy of config file into longname directory immediately (if you wait it may have been modified before
        stacking is complete)

        :param fp_in: path to config file.
        :param overwrite_results: Overwrite existing if True.
        '''

        if 'shortname' in self.config_dict['io']:
            shortname = self.config_dict['io']['shortname']
        else:
            shortname = os.path.basename(fp_in).split('.')[0]

        longname = self.construct_longname(shortname)

        out_file_path = os.path.join(self.parse_path(self.config_dict['io']['output_folder']), longname)

        if not os.path.exists(out_file_path):
            os.makedirs(out_file_path)
        else:
            if not overwrite_results:
                while os.path.exists(out_file_path):
                    out_file_path = out_file_path + "_"
                os.makedirs(out_file_path)
        self.config_dict['io']['saved_data_path'] = out_file_path

        # Copy Config File
        fp_name = os.path.basename(fp_in)
        fp_out = os.path.join(out_file_path, fp_name)
        logging.info("Copying parameter file...")
        logging.info("  FROM : {}".format(fp_in))
        logging.info("    TO : {}".format(fp_out))
        logging.info("")
        shutil.copyfile(fp_in, fp_out)
        self.config_dict['io']['config_ini'] = fp_out

    def construct_longname(self, basename):
        ''' Use parameters in config file to create a "longname" which directories and files are named.

        :param basename: customizable "shortname" that proceeds the generic longname.
        :return longname: name used for directory and pickle file.
        '''
        longname = "_".join([basename,])
        self.config_dict['io']['longname'] = longname
        return longname

    def parse_path(self, path_in):

        path_in = path_in.split(" ")
        if len(path_in) == 1:
            return path_in[0]
        else:
            path_env = os.environ[path_in[0]]
            if len(path_in) == 2:
                if 'nt' in os.name:
                    return path_env + os.path.join('\\', path_in[1].replace('/', '\\'))
                else:
                    return path_env + os.path.join('/', path_in[1])
            else:
                if 'nt' in os.name:
                    path_rename = [i.replace('/', '\\') for i in path_in[1:]]
                    return path_env + os.path.join('\\', *path_rename)
                else:
                    return path_env + os.path.join('/', *path_in[1:])

    @staticmethod
    def beam_solid_angle_from_fwhm(fwhm):
        sig = (fwhm / 3600 * np.pi / 180) / 2.355
        omega_beam = 2 * np.pi * sig ** 2
        return omega_beam

    @staticmethod
    def k_to_ell(k):
        # ell = 2pi k_theta
        ell = k * 2 * np.pi / (np.pi / 180 / 60)
        return ell

    @staticmethod
    def ell_to_k(ell):
        # ell = 2pi k_theta
        k_theta = ell / 2 / np.pi * (np.pi / 180 / 60)
        return k_theta

    @staticmethod
    def import_saved_pickles(pickle_fn):
        with open(pickle_fn, "rb") as file_path:
            encoding = pickle.load(file_path)
        return encoding

    @staticmethod
    def save_to_pickles(save_path, save_file):
        with open(save_path, "wb") as pickle_file_path:
            pickle.dump(save_file, pickle_file_path)

    @staticmethod
    def lambda_to_ghz(lam):
        c_light = 299792458.0  # m/s
        return np.array([1e-9 * c_light / (i * 1e-6) for i in lam])

    @staticmethod
    def get_psf_correction_linear(ell, fwhm_arcsec, fwhm_arcsec2=None):
        sigma_radian1 = (fwhm_arcsec / 3600 * np.pi / 180) / np.sqrt(8 * np.log10(2))
        sigma_ell1 = 1 / sigma_radian1
        Bl1 = np.exp(-0.5 * (ell / sigma_ell1) ** 2)
        if fwhm_arcsec2:
            sigma_radian2 = (fwhm_arcsec2 / 3600 * np.pi / 180) / np.sqrt(8 * np.log10(2))
            sigma_ell2 = 1 / sigma_radian2
            Bl2 = np.exp(-0.5 * (ell / sigma_ell2) ** 2)
            Bl = np.sqrt(Bl1 * Bl2)
            return Bl
        else:
            return Bl1

    @staticmethod
    def get_weighted_Bl(ell_lo, deltal, sigma_radian1):
        sigma_ell1 = 1 / sigma_radian1
        ell = np.arange(ell_lo, ell_lo + deltal)
        iBl = np.sum((2 * ell + 1) * (np.exp(-0.5 * (ell / sigma_ell1) ** 2))) / np.sum((2 * ell + 1))
        return iBl

    @staticmethod
    def get_psf_correction(ell_bins, fwhm_arcsec, fwhm_arcsec2=None):
        sigma_radian1 = (fwhm_arcsec / 3600 * np.pi / 180) / np.sqrt(8 * np.log10(2))
        Bl1 = np.array([Toolbox.get_weighted_Bl(ell_bins[i], d, sigma_radian1)
                        for i, d in enumerate(np.diff(ell_bins))])

        if fwhm_arcsec2:
            sigma_radian2 = (fwhm_arcsec2 / 3600 * np.pi / 180) / np.sqrt(8 * np.log10(2))
            Bl2 = np.array([Toolbox.get_weighted_Bl(ell_bins[i], d, sigma_radian2)
                            for i, d in enumerate(np.diff(ell_bins))])
            Bl = np.sqrt(Bl1 * Bl2)
            return Bl
        else:
            return Bl1

    @staticmethod
    def pixel_beam_function_linear(ell, pix_arcsec):

        # from: http: // xxx.lanl.gov / abs / astro - ph / 0007212
        # calculate pixel beam function
        # for square pixels res_arcmin is the side of the square pixel

        #pix = res_arcmin / 60 * dtor
        pix = pix_arcsec / 3600 * dtor

        out = (np.exp(-1 / 18.1 * (ell * pix) ** 2.04) * (1 - 0.0272 * (ell * pix) ** 2))

        return out**2

    @staticmethod
    def get_weighted_pixel(ell_lo, deltal, pix):
        ell = np.arange(ell_lo, ell_lo + deltal)
        out = np.sum((2 * ell + 1) * (np.exp(-1 / 18.1 * (ell * pix) ** 2.04) * (1 - 0.0272 * (ell * pix) ** 2))) / \
              np.sum((2 * ell + 1))
        return out

    @staticmethod
    def pixel_beam_function(ell_bins, pix_arcsec):

        # from: http: // xxx.lanl.gov / abs / astro - ph / 0007212
        # calculate pixel beam function
        # for square pixels res_arcmin is the side of the square pixel

        pix = pix_arcsec / 3600 * dtor
        out = np.array([Toolbox.get_weighted_pixel(ell_bins[i], d, pix)
                        for i, d in enumerate(np.diff(ell_bins))])

        return out**2

    def comoving_volume_given_area(self, area_deg2, zz1, zz2):
        vol0 = self.config_dict['cosmology_dict']['cosmology'].comoving_volume(zz2) - \
               self.config_dict['cosmology_dict']['cosmology'].comoving_volume(zz1)
        vol = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi) * vol0
        return vol

    def moster2011_cosmic_variance(self, z, dz=0.2, field='cosmos'):
        cv_params = {'cosmos': [0.069, -.234, 0.834], 'udf': [0.251, 0.364, 0.358]
            , 'goods': [0.261, 0.854, 0.684], 'gems': [0.161, 0.520, 0.729]
            , 'egs': [0.128, 0.383, 0.673]}

        field_params = cv_params[field]
        sigma_cv_ref = field_params[0] / (z ** field_params[2] + field_params[1])

        if dz == 0.2:
            sigma_cv = sigma_cv_ref
        else:
            sigma_cv = sigma_cv_ref * (dz / 0.2) ** (-0.5)

        return sigma_cv

    def clean_nans(self, dirty_array, replacement_char=0.0):
        clean_array = dirty_array.copy()
        clean_array[np.isnan(dirty_array)] = replacement_char
        clean_array[np.isinf(dirty_array)] = replacement_char

        return clean_array

    def gauss(self, x, x0, y0, sigma):
        p = [x0, y0, sigma]
        return p[1] * np.exp(-((x - p[0]) / p[2]) ** 2)

    def gauss_kern(self, fwhm, side, pixsize):
        ''' Create a 2D Gaussian (size= side x side)'''

        sig = fwhm / 2.355 / pixsize
        delt = np.zeros([int(side), int(side)])
        delt[0, 0] = 1.0
        ms = np.shape(delt)
        delt = self.shift_twod(delt, ms[0] / 2, ms[1] / 2)
        kern = delt
        gaussian_filter(delt, sig, output=kern)
        kern /= np.max(kern)

        return kern

    @staticmethod
    def shift_twod(seq, x, y):
        out = np.roll(np.roll(seq, int(x), axis=1), int(y), axis=0)
        return out

    def smooth_psf(self, mapin, psfin):

        s = np.shape(mapin)
        mnx = s[0]
        mny = s[1]

        s = np.shape(psfin)
        pnx = s[0]
        pny = s[1]

        psf_x0 = pnx / 2
        psf_y0 = pny / 2
        psf = psfin
        px0 = psf_x0
        py0 = psf_y0

        # pad psf
        psfpad = np.zeros([mnx, mny])
        psfpad[0:pnx, 0:pny] = psf

        # shift psf so that centre is at (0,0)
        psfpad = self.shift_twod(psfpad, -px0, -py0)
        smmap = np.real(np.fft.ifft2(np.fft.fft2(mapin) *
                                     np.fft.fft2(psfpad))
                        )

        return smmap

    def dist_idl(self, n1, m1=None):
        ''' Copy of IDL's dist.pro
        Create a rectangular array in which each element is
        proportinal to its frequency'''

        if m1 == None:
            m1 = int(n1)

        x = np.arange(float(n1))
        for i in range(len(x)): x[i] = min(x[i], (n1 - x[i])) ** 2.

        a = np.zeros([int(n1), int(m1)])

        i2 = m1 // 2 + 1

        for i in range(i2):
            y = np.sqrt(x + i ** 2.)
            a[:, i] = y
            if i != 0:
                a[:, m1 - i] = y

        return a

    def circle_mask(self, pixmap, radius_in, pixres):
        ''' Makes a 2D circular image of zeros and ones'''

        radius = radius_in / pixres
        xy = np.shape(pixmap)
        xx = xy[0]
        yy = xy[1]
        beforex = np.log2(xx)
        beforey = np.log2(yy)
        if beforex != beforey:
            if beforex > beforey:
                before = beforex
            else:
                before = beforey
        else:
            before = beforey
        l2 = np.ceil(before)
        pad_side = int(2.0 ** l2)
        outmap = np.zeros([pad_side, pad_side])
        outmap[:xx, :yy] = pixmap

        dist_array = self.shift_twod(self.dist_idl(pad_side, pad_side), pad_side / 2, pad_side / 2)
        circ = np.zeros([pad_side, pad_side])
        ind_one = np.where(dist_array <= radius)
        circ[ind_one] = 1.
        mask = np.real(np.fft.ifft2(np.fft.fft2(circ) *
                                    np.fft.fft2(outmap))
                       ) * pad_side * pad_side
        mask = np.round(mask)
        ind_holes = np.where(mask >= 1.0)
        mask = mask * 0.
        mask[ind_holes] = 1.
        maskout = self.shift_twod(mask, pad_side / 2, pad_side / 2)

        return maskout[:xx, :yy]

    def map_rms(self, map, mask=None):
        if mask != None:
            ind = np.where((mask == 1) & (self.clean_nans(map) != 0))
            print('using mask')
        else:
            ind = self.clean_nans(map) != 0
        map /= np.max(map)

        x0 = abs(np.percentile(map, 99))
        hist, bin_edges = np.histogram(np.unique(map), range=(-x0, x0), bins=30, density=True)

        p0 = [0., 1., x0 / 3]
        x = .5 * (bin_edges[:-1] + bin_edges[1:])
        x_peak = 1 + np.where((hist - max(hist)) ** 2 < 0.01)[0][0]

        # Fit the data with the function
        fit, tmp = curve_fit(self.gauss, x[:x_peak], hist[:x_peak] / max(hist), p0=p0)
        rms_1sig = abs(fit[2])

        return rms_1sig

    def leja_mass_function(self, z, Mass=np.linspace(9, 13, 100), sfg=2):
        # sfg = 0  -  Quiescent
        # sfg = 1  -  Star Forming
        # sfg = 2  -  All

        nz = np.shape(z)

        a1 = [-0.10, -0.97, -0.39]
        a2 = [-1.69, -1.58, -1.53]
        p1a = [-2.51, -2.88, -2.46]
        p1b = [-0.33, 0.11, 0.07]
        p1c = [-0.07, -0.31, -0.28]
        p2a = [-3.54, -3.48, -3.11]
        p2b = [-2.31, 0.07, -0.18]
        p2c = [0.73, -0.11, -0.03]
        ma = [10.70, 10.67, 10.72]
        mb = [0.00, -0.02, -0.13]
        mc = [0.00, 0.10, 0.11]

        aone = a1[sfg] + np.zeros(nz)
        atwo = a2[sfg] + np.zeros(nz)
        phione = 10 ** (p1a[sfg] + p1b[sfg] * z + p1c[sfg] * z ** 2)
        phitwo = 10 ** (p2a[sfg] + p2b[sfg] * z + p2c[sfg] * z ** 2)
        mstar = ma[sfg] + mb[sfg] * z + mc[sfg] * z ** 2

        # P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        P = np.array([aone, mstar, phione, atwo, mstar, phitwo])
        return self.dschecter(Mass, P)

    def schecter(self, X, P, exp=None, plaw=None):
        ''' X is alog10(M)
            P[0]=alpha, P[1]=M*, P[2]=phi*
            the output is in units of [Mpc^-3 dex^-1] ???
        '''
        if exp != None:
            return np.log(10.) * P[2] * np.exp(-10 ** (X - P[1]))
        if plaw != None:
            return np.log(10.) * P[2] * (10 ** ((X - P[1]) * (1 + P[0])))
        return np.log(10.) * P[2] * (10. ** ((X - P[1]) * (1.0 + P[0]))) * np.exp(-10. ** (X - P[1]))

    def dschecter(self, X, P):
        '''Fits a double Schechter function but using the same M*
           X is alog10(M)
           P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        '''
        rsch1 = np.log(10.) * P[2] * (10. ** ((X - P[1]) * (1 + P[0]))) * np.exp(-10. ** (X - P[1]))
        rsch2 = np.log(10.) * P[5] * (10. ** ((X - P[4]) * (1 + P[3]))) * np.exp(-10. ** (X - P[4]))

        return rsch1 + rsch2

    def loggen(self, minval, maxval, npoints, linear=None):
        points = np.arange(npoints) / (npoints - 1)
        if (linear != None):
            return (maxval - minval) * points + minval
        else:
            return 10.0 ** ((np.log10(maxval / minval)) * points + np.log10(minval))

    def linloggen(self, deltal, width, minval=None, maxval=None, ltrans=None, verbose=False):

        if not minval:
            minval = deltal / 2.
        if not maxval:
            maxval = self.k_to_ell(4.)
        if verbose:
            print('minval={0:0.1f}'.format(minval))
            print('maxval={0:0.1f}'.format(maxval))
        nlin = int(np.floor(1. / width))
        if verbose:
            print('nlin={0:0.1f}'.format(nlin))
        ltrans = nlin * deltal + minval
        if verbose:
            print('ltrans={0:0.1f}'.format(ltrans))

        npoints = 2 * np.log10(maxval / ltrans) / width
        if verbose:
            print('npoints={0:0.3f}'.format(npoints))

        points = np.arange(np.floor(npoints)) / (npoints)
        if verbose:
            print('len(points)={0:0.3f}'.format(len(points)))

        n = int(np.floor(npoints + nlin))
        if verbose:
            print('n={0:0.3f}'.format(n))

        ell = np.zeros(n)
        ell[:nlin] = np.arange(nlin) * deltal + minval
        ell[nlin:] = 10 ** ((np.log10(maxval / ltrans)) * points + np.log10(ltrans))

        return ell
