###############################################################################
# Main Program : cmb_extract.py                                               #
# Sub-Program  : cmb_extract_config.py                                        #
# Description  : Parameter file for controlling options of Main Program       #
#                                                                             #
# Current Version : V17                                                       #
#                                                                             #
# - Supports Planck CMB Skymaps (Full Mission) in FITS format from the ESA    #
#   (European Space Agency) - http://pla.esac.esa.int/pla/#maps               #
# - Supports Simulated non-Gaussian CMB maps in FITS format from the GAVO     #
#   (German Astrophysical Virtual Observatory)                                #
#  - http://dc.zah.uni-heidelberg.de/elsnersim/q/s/fixed                      #
#                                                                             #
# - Uses python3                                                              #
#   - healpy (manipulating HEALpix formatted data from FITS files             #
#     - independently specified Nside per input file                          #
#     - positive and negative injection of Simulated non-Gaussianity          #
#     - Gudhi TDA output                                                      #
#     - configurable sample output size                                       #
#     - configurable CMB temperature scaling factor                           #
#     - configurable pixel sampling function                                  #
#                                                                             #
###############################################################################

import numpy as np

SYSTEM_LOC = '../'
DATA_LOC = f'{SYSTEM_LOC}/data'
OUTPUT_LOC = f'{SYSTEM_LOC}/output'
SKY_MAP_LOC = f'{DATA_LOC}/sky_maps'
SIM_MAP_LOC = f'{DATA_LOC}/simulated_maps'

# Pickle input/output files
save_CMB_sample = 'saved_CMB_sample.pickle'
save_CMB_homology = 'saved_CMB_homology.pickle'

# Maximum Dimension (Rips Complexes)
max_dim = 3
patch_pixel_width=32
true_norm_false_product=True
# Alpha Complex (controls Max Dimensions)
# Unused - not using Alpha Complex
max_alpha_square = 2

# full sky map location (galactic plane removed)
# found in SKY_MAP_LOC above
sky_map_1 ='COM_CMB_IQU-commander_2048_R3.00_full.fits'
sky_map_2 ='COM_CMB_IQU-smica_2048_R3.00_full.fits'
sky_mask = 'COM_Mask_CMB-Inpainting-Mask-Int_2048_R3.00.fits'

# hard-coded nside of Sky Map (for SKY_MAP above)
sky_map_1_nside = 2048
sky_map_2_nside = 2048

# unused for now - degrade sky maps to common nside
sky_map_adj_nside = 2048

# simulated Gaussian map
# no nside required
sim_map = 'alm_l_0100_v3.fits'

# simulated Non-Gaussian map
# no nside required
sim_map_ng = 'alm_nl_0100_v3.fits'

# hard-coded sample size
num_samples = 1000

# number of times to repeat for sim_map_fnl_set 
num_repeats = 5

# levels of non-gaussianity (NG) to inject
# sim_map_fnl_set = [0, 100]
sim_map_fnl_set = [0, 100]
# sim_map_fnl_set = [-10, -20, 0, 10, 20, 100]

# multiple nsides
sim_map_nside = 2048

# list of Temperature Dimensions for correlations
num_temperatures = 4

# increase if sampling fails to find points above threshold
max_attempts = 1e8

# number of standard deviations of Temperature to consider
std_factor = 1

# Temperature Scale Factor (1 = actual value, 10000 = actual value * 10000)
temperature_scale_factor = 1e5

# Take absolute value of temperatures
take_absolute_temperature=True

# Filter above or below threshold
above_threshhold=True

# Pixel Sampling
pixel_overlap = True

# 0.7 degrees < pixel angle < 2.2 degrees
max_radius = np.radians(2.2)
min_radius = np.radians(0.07)

# pixels within bins
min_bin_points = 50

# full sky map location
full_path_sky_map_1 = f'{SKY_MAP_LOC}/{sky_map_1}'
full_path_sky_map_2 = f'{SKY_MAP_LOC}/{sky_map_2}'
full_path_sky_mask = f'{SKY_MAP_LOC}/{sky_mask}'

# Please cite http://adsabs.harvard.edu/abs/2009ApJS..184..264E
# for using Simulated Maps sim_gauss and sim_non_gauss below
# FITS file of spherical harmonic coefficients in HEALPix format
# NSIDE is undefined for spherical harmonic coefficients (alm)
# multipole moment l_max=1024
# l_max determines number alm coefficients = length of alm array
# length of alm array = m_max(2*l_max + 1 - m_max)/2 + l_max + 1 = 525 825

# simulated map
full_path_sim_map  = f'{SIM_MAP_LOC}/{sim_map}'

# simulated non-Gaussian map
full_path_sim_map_ng  = f'{SIM_MAP_LOC}/{sim_map_ng}'

##################################
from sklearn.base          import BaseEstimator, TransformerMixin

class MyPadding(BaseEstimator, TransformerMixin):
    """
    This is a class for padding a list of persistence diagrams with dummy points, so that all persistence diagrams end up with the same number of points.
    """
    
    def __init__(self, use=False):
        """
        Constructor for the Padding class.

        Parameters:
            use (bool): whether to use the class or not (default False).
        """
        self.use = use

    def fit(self, X, y=None):
        """
        Fit the Padding class on a list of persistence diagrams (this function actually does nothing but is useful when Padding is included in a scikit-learn Pipeline).

        Parameters:
            X (list of n x 2 or n x 1 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        self.max_pts = max([len(diag) for diag in X])
        return self
    
    def transform(self, X):
        """
        Add dummy points to each persistence diagram so that they all have the same cardinality. All points are given an additional coordinate indicating if the point was added after padding (0) or already present before (1).  

        Parameters:
            X (list of n x 2 or n x 1 numpy arrays): input persistence diagrams.

        Returns:
            list of n x 3 or n x 2 numpy arrays: padded persistence diagrams.
        """
        if self.use:
            Xfit, num_diag = [], len(X)
            for diag in X:
                diag_pad = np.pad(diag, ((0,max(0, self.max_pts - diag.shape[0])), (0,0)), "constant", constant_values=((0,0),(0,0)))
                #diag_pad[:diag.shape[0],2] = np.ones(diag.shape[0])
                Xfit.append(diag_pad)                    
        else:
            Xfit = X
        return Xfit

    def __call__(self, diag):
        """
        Apply Padding on a single persistence diagram and outputs the result.

        Parameters:
            diag (n x 2 numpy array): input persistence diagram.

        Returns:
            n x 2 numpy array: padded persistence diagram.
        """
        return self.fit_transform([diag])[0]
