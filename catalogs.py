import pdb
import os
import json
import numpy as np
from astropy.io import fits

class Catalogs:
	'''
	This Class creates a catalog object containing the raw catalog and a split_table
	for which each category is split into indices, for example, if you have three redshift
	bins [1,2,3], the "redshift" column of split_dict will contain 0,1,2, and nans, where
	each object between z=1 and 2 has value 0; between z=1 and 2 has value 1... and objects
	outside that range have values nan.  Same is true for bins of "stellar_mass", "magnitude",
	or however the catalog is split.

	Required parameters are defined in the config.ini file
	- path (str); format CATSPATH PATH
	- file (str); format FILENAME
	- astrometry (dict); contains "ra", "dec", which indicate column names for positions in RA/DEC.
	- classification (dict); contains dicts for "split_type", "redshift", and optional categories, e.g.,
		"split_params", "stellar_mass", "magnitudes", etc. Specifically:
		-- "split_type" can be "label", "uvj", "nuvrj"
			> "label" is ..
			> "uvj" is ...
			> "nuvrj" is ..
		-- "split_params" can be
			> "labels"
		--
	'''

	catalog_dict = {}

	def __init__(self):
		pass

	def import_catalog(self, mask=None):
		''' Import a CSV file and split into populations.

		:param keep_catalog: If True stores a copy of catalog in self (large file!)
		:param qg_zcut: The redshift cutoff for application of star-forming/quiescent split.
		'''

		catalog_params = self.config_dict['catalog']
		layers_dir = catalog_params['layers_dir']
		path_catalog = os.path.join(self.parse_path(catalog_params['path']), layers_dir)

		if os.path.isdir(path_catalog):
			path_files = os.listdir(path_catalog)
			#self.catalog_dict['files'] = {layers_dir: path_files}
			#fwhm_files = np.unique([i.split('fwhm_')[-1].split('.')[0] for i in path_files])
			#map_key_dict = {'PSW': '__fwhm_17p6', 'PMW': '__fwhm_23p9', 'PLW': '__fwhm_35p2'}
			#fwhm_key_dict = {'17p6': 'PSW', '23p9': 'PMW', '35p2': 'PLW'}
			beam_area_key = {'17p6': 1.079e-8, '23p9': 1.079e-8, '35p2': 3.877e-8}
			color_correction_key = {'17p6': 1.047, '23p9': 1.035, '35p2': 1.038}
			wavelength_key = {'17p6': 250, '23p9': 350, '35p2': 500}
			#mapname_key = {'17p6': 250, '23p9': 350, '35p2': 500}
			mapname_key = {'17p6': 'PSW', '23p9': 'PMW', '35p2': 'PLW'}

			for ifile in path_files:
				if 'DS_Store' not in ifile:
					map_params = {}
					ikey = ifile.split('fwhm_')[-1].split('.')[0]
					map_params['path_map'] = os.path.join(path_catalog, ifile)
					#print(wavelength_key, ikey)
					#pdb.set_trace()
					map_params['wavelength'] = wavelength_key[ikey]
					map_params['beam'] = {"fwhm": float(ikey.replace('p','.')), "area": beam_area_key[ikey]}
					map_params['color_correction'] = color_correction_key[ikey]
					if 'mask' in catalog_params:
						mapname = mapname_key[ikey]
						for imask in self.config_dict['maps']:
							if mapname.lower() in imask.lower():
								basename = os.path.basename(self.config_dict['maps'][imask]['path_map']).split('.')[0]
								footprint_path = os.path.join(self.parse_path(catalog_params['mask']), basename)+\
												 '_footprint.fits'
					else:
						footprint_path=None

					map_dict = self.import_map_dict(map_params,
													zeropad=False,
													mask=mask,
													footprint_path=footprint_path)
					self.catalog_dict[ifile.split('.')[0]] = map_dict

			#fwhm = [float(i.split('fwhm_')[-1].split('.')[0].replace('p', '.')) for i in path_files]
			#map_key = [fwhm_key_dict[i.split('fwhm_')[-1].split('.')[0]] for i in path_files]

			#self.catalog_dict['catalog']['maps'] = {key: {} for key in map_key_dict}
			#for i, ifilename in enumerate(path_files):
			#	self.catalog_dict['catalog']['maps'][map_key[i]][ifilename] = 44

		else:
			print("Catalog directory not found: "+path_catalog)

		#pdb.set_trace()
