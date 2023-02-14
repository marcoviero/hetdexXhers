import pdb

from pseudospectrum import PseudoSpectrum

class PseudoSpectrumWrapper(PseudoSpectrum):
    ''' SimstackWrapper consolidates each step required to stack:
        - Read in parameters from config.ini file
        - Read in catalogs
        - Read in maps
        - Split catalog into bins specified in config file
        - Run stacking algorithm, which includes:
            -- create convolved layer cube at each wavelength [and optionally redshift]
        - Parse results into user-friendly pandas DataFrames

        :param param_path_file: (str)  path to the config.ini file
        :param read_maps: (bool) [optional; default True] If prefer to do this manually then set False
        :param read_catalogs: (bool) [optional; default True] If prefer to do this manually then set False
        :param stack_automatically: (bool) [optional; default False] If prefer to do this automatically then set True

        TODO:
        - counts in bins
        - agn selection to pops
        - CIB estimates
        - Simulated map of best-fits

    '''
    def __init__(self,
                 param_file_path,
                 read_maps=False,
                 mask=None,
                 read_catalog=False,
                 save_automatically=True,
                 overwrite_results=False):

        super().__init__(param_file_path)

        # Copy configuration file immediately, before it can be changed.
        if save_automatically:
            self.copy_config_file(param_file_path, overwrite_results=overwrite_results)

        if read_catalog:
            self.import_catalog(mask=mask)  # This happens in catalogs.py

        if read_maps:
            self.import_maps(mask=mask)  # This happens in maps.py