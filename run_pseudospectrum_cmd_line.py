#!/usr/bin/env python

from pseudospectrumwrapper import PseudoSpectrumWrapper

# Standard modules
import os
import pdb
import sys
import time
import logging

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%d-%m %I:%M:%S %p')

    # Get parameters from the provided parameter file
    if len(sys.argv) > 1:
        param_file_path = sys.argv[1]
    else:
        #param_file_path = os.path.join('config', 'hetdexXhers.ini')
        #param_file_path = os.path.join('hetdexXhers.ini')
        param_file_path = os.path.join('sptXspt.ini')

    # Instantiate SIMSTACK object
    hetdexXhers_object = PseudoSpectrumWrapper(param_file_path,
                                               read_maps=True,
                                               read_catalog=False,
                                               save_automatically=False,
                                               )

    hetdexXhers_object.copy_config_file(param_file_path, overwrite_results=False)
    print('Now Cross-Correlating', param_file_path)
    t0 = time.time()

    # Import maps and masks.
    #hetdexXhers_object.import_maps(hanning=True)

    # Cross-correlate according to parameters in parameter file
    #pdb.set_trace()

    # Define auto-only or auto and cross-spectra.
    cross_spectra = False

    # Create MKK
    mkk_dict = hetdexXhers_object.get_mkks(cross_spectra=cross_spectra,
                                           iterations=20)

    pdb.set_trace()
    # Perform Cross Correlation
    hetdexXhers_object.get_pseudospectra(cross_spectra=cross_spectra)

    # Save Results
    #saved_pickle_path = hetdexXhers_object.save_pseudospectra(param_file_path)

    # Summarize timing
    t1 = time.time()
    tpass = t1 - t0

    logging.info("Cross Spectra Successful!")
    logging.info("Find Results in {}".format(saved_pickle_path))
    logging.info("")
    logging.info("Total time  : {:.4f} minutes\n".format(tpass / 60.))

if __name__ == "__main__":
    main()
else:
    logging.info("Note: `mapit` module not being run as main executable.")