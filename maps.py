import pdb
import os
import json
import numpy as np
from astropy.io import fits

class Maps:
	'''
	This Class creates objects for a set of
	maps/noisemaps/beams/etc., at each Wavelength.

	Each map is defined through parameters in the config.ini file:
	- wavelength (float)
	- beam (dict); contains "fwhm" in arcsec and "area" in steradian**2.
	- color_correction (float)
	- path_map (str)
	- path_noise (str)
	'''

	maps_dict = {}

	def __init__(self):
		pass

	def import_maps(self, zeropad=False, mask=None):
		''' Import maps (and optionally noisemaps) described in config file.

		:return: Map dictionary stored in self.maps_dict.
		'''

		for imap in self.config_dict['maps']:
			print(imap)
			try:
				map_params = self.config_dict['maps'][imap].copy()
			except:
				print('failed to read')
				pass
				#pdb.set_trace()
			map_dict = self.import_map_dict(map_params,
											zeropad=zeropad,
											mask=mask)
			self.maps_dict[imap] = map_dict

	def import_map_dict(self, map_params, zeropad=False, mask=None):
		''' Import maps described in config file and populate map_dict with parameters.

		:param map_dict:
		:return: populated map_dict
		'''
		#pdb.set_trace()
		file_map = self.parse_path(map_params["path_map"])
		if 'path_noise' in map_params:
			file_noise = self.parse_path(map_params["path_noise"])
		else:
			file_noise = None
		wavelength = map_params["wavelength"]
		psf = map_params["beam"]
		beam_area = map_params["beam"]["area"]
		color_correction = map_params["color_correction"]

		#SPIRE Maps have Noise maps in the second extension.
		if file_noise is not None:
			if file_map == file_noise:
				header_ext_map = 1
				header_ext_noise = 2
			else:
				header_ext_map = 0
				header_ext_noise = 0
		else:
			header_ext_map = 0
			header_ext_noise = None

		if not os.path.isfile(file_map):
			file_map = os.path.join('..', file_map)
			try:
				file_noise = os.path.join('..', file_noise)
			except:
				file_noise = None

		if os.path.isfile(file_map):
			try:
				cmap, hd = fits.getdata(file_map, header_ext_map, header=True)
			except:
				cmap, hd = fits.getdata(file_map, 1, header=True)
			if file_noise is not None:
				try:
					cnoise, nhd = fits.getdata(file_noise, header_ext_noise, header=True)
				except:
					cnoise, nhd = fits.getdata(file_noise, 1, header=True)
		else:
			print("Files not found, check path in config file: "+file_map)
			pdb.set_trace()

		#GET MAP PIXEL SIZE
		dims = np.shape(cmap)
		if 'CD2_2' in hd:
			pix = hd['CD2_2'] * 3600.
		else:
			pix = hd['CDELT2'] * 3600.

		#GET MASKS
		mask_dict = {}
		mask_dict['mask'] = np.ones_like(cmap)
		mask_dict['mask'][np.isnan(cmap)] = 0
		if mask is not None:
			if type(mask) is dict:
				if 'kaiser' in mask:
					mask_dict['kaiser'] = np.sqrt(
						np.outer(np.abs(np.kaiser(dims[0], mask['kaiser'])),
								 np.abs(np.kaiser(dims[1], mask['kaiser']))))
			if 'hanning' in mask:
				mask_dict['hanning'] = np.sqrt(np.outer(np.abs(np.hanning(dims[0])), np.abs(np.hanning(dims[1]))))
			if 'blackman' in mask:
				mask_dict['blackman'] = np.sqrt(np.outer(np.abs(np.blackman(dims[0])), np.abs(np.blackman(dims[1]))))

		map_dict = {}
		map_dict['masks'] = mask_dict

		#GET KMAP
		#map_dict['kmap'] = self.get_kmap(dims, pix)
		map_dict['kmap'] = self.get_k_from_map(cmap, pix)


		#READ BEAMS
		fwhm = psf["fwhm"]
		#print(fwhm)
		kern = self.gauss_kern(fwhm, np.floor(fwhm * 8.)/pix, pix)

		map_dict["map"] = self.clean_nans(cmap) * color_correction
		if beam_area != 1.0:
			map_dict["map"] /= beam_area #* 1e6
		if file_noise is not None:
			map_dict["noise"] = self.clean_nans(cnoise, replacement_char=1e10) * color_correction
			if beam_area != 1.0:
				map_dict["noise"] /= beam_area #* 1e6

		map_dict["header"] = hd
		map_dict["pixel_size"] = pix
		map_dict["psf"] = self.clean_nans(kern)

		if wavelength != None:
			map_dict["wavelength"] = wavelength

		if fwhm != None:
			map_dict["fwhm"] = fwhm

		return map_dict
