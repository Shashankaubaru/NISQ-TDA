import configcmb
from configcmb import MyPadding

import ray
from datetime import datetime 

import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import sys
import time
# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=DeprecationWarning)
#     # trying to stop np.int deprecation warning in test, but to no avail
import gudhi as gd
import gudhi.representations as gr
import healpy as hp
import argparse

from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

# from sklearn.metrics import roc_curve as roc
# from sklearn.metrics import roc_auc_score as roc_auc_score
from sklearn.metrics import auc as auc
from sklearn.metrics import roc_curve as roc_curve
import pickle

from tqdm import tqdm
import itertools
from numpy.lib import recfunctions as rfn
import os
from os import path
# import pdb

from scipy.spatial.transform import Rotation as R

from bayes_tda.classifiers import EmpBayesFactorClassifier as EBFC
from bayes_tda.intensities import RGaussianMixture
from bayes_tda.intensities import Posterior
# from bayes_tda.bccfcc_post import _show_bayes_factor_distribution # made own locally

EPS=1e-16

###################################################
# MAP HELPER FUNCTIONS
###################################################

def gen_pairs(pixel_list):
    """Return all combinations of pairs from list, as iterator."""
    return itertools.combinations(pixel_list, 2)

def gen_random_pixel_ids(num_samples, nside):

    npix = hp.nside2npix(nside)
    rng = np.random.default_rng()
    random_samples = rng.integers(npix, size=num_samples)

    return random_samples

def query_disc(nside, cpixel, radius=configcmb.max_radius):

    return hp.query_disc(nside, hp.pix2vec(nside, cpixel),
                         radius=radius,
                         inclusive=configcmb.pixel_overlap)

# Read Field from CMB Map (FITS file HEALPIX Format)- all pixels
def read_CMB_map(sky_file, norm_by_var=True):

    # read CMB file - temperature (field[0])
    hp.disable_warnings()
    sky_map = hp.read_map(sky_file, field=0, dtype=np.float64, partial=False,
                            verbose=False)
    if norm_by_var:
        sky_map = sky_map/np.sqrt(np.var(sky_map))
    else:
        sky_map = sky_map*configcmb.temperature_scale_factor
    
    return sky_map


def make_map_from_alm(nside, alm_gauss_file, alm_non_gauss_file=None, fnl=None, norm_by_var=True):

    alm_gauss, alm_gauss_mmax = hp.read_alm(alm_gauss_file, hdu=1, return_mmax=True)

    if fnl is None:
        sky_map = hp.alm2map(alm_gauss, nside,
                             fwhm=0, sigma=None, pol=False, verbose=False)
    else:
        alm_non_gauss, alm_non_gauss_mmax = hp.read_alm(
            alm_non_gauss_file, hdu=1, return_mmax=True)

        if not (alm_gauss_mmax == alm_non_gauss_mmax):
            raise(ValueError("Input alm files, differ in size"))
        
        sky_map = hp.alm2map(alm_gauss + fnl * alm_non_gauss, nside,
                             fwhm=0, sigma=None, pol=False, verbose=False) # RING ordering

    if norm_by_var:
        sky_map = sky_map/np.sqrt(np.var(sky_map))
    else:
        sky_map = sky_map*configcmb.temperature_scale_factor
                  
    return sky_map


def calc_alpha_angle(v1, v2):  # assumming normalized
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

def len_of_lists_in_list(options_list):
    return list(map((lambda innerlist: len(innerlist)), options_list))

#################################################
# VISUALISATIONS
#################################################

def visualize_point_set(points):
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')

    ax.scatter(points[:,0], points[:,1], points[:,2]) #marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plot.show()

def visualize_sample_fnls(sample_fnls):

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(2):
        ax.scatter(sample_fnls[0,i][:,0],
                   sample_fnls[0,i][:,1],
                   sample_fnls[0,i][:,2],
                   color=['red','green'][i],
                   marker=['o','x'][i])
    plot.show()

def visualize_surface(patch):
    n = len(patch)
    x_cords, y_cords = np.indices((n, n))
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_cords, y_cords, patch)
    fig.show()

def visualize_grid(grid, nside=1024):
    points = np.array([hp.pix2vec(nside, pixel) for row in grid for pixel in row])
    visualize_point_set(points)

def make_string_from_options(collect_options, fnl_0, fnl_1):
    
    return '-'.join(
        ['f0-' + str(fnl_0) + '-f1-' + str(fnl_1)]+
        list(map(lambda opt: str(opt[0])+ '' + str(opt[1]),
                 list(zip(list(collect_options.keys()), list(collect_options.values()))))))

def make_text_from_options(**collect_options):
    return '\n'.join(
        list(map(lambda opt: str(opt[0])+ ': ' + str(opt[1]),
                 list(zip(list(collect_options.keys()), list(collect_options.values())))))) + "\n"


#################################################
# DATA WRITING AND LOADING
#################################################

def load_sample_data(file_name):

   print("\nReading Pickle file = ", file_name)

   print("Loading CMB Sample data\n", flush=True)

   sample_fnls = pickle.load(open(file_name, 'rb'))

   print("CMB Sample load complete\n", flush=True)

   return sample_fnls


def load_homology_data(file_name):

   print("\nReading Pickle file = ", args.f)

   print("Loading CMB Homology data\n", flush=True)

   data_PIs, data_labels = \
             pickle.load(open(file_name, 'rb'))

   print("CMB Homology load complete\n", flush=True)

   return data_PIs, data_labels


def find_bounding_box_for_PI(data_PIs_betti):

    max_range = 1<<20
    xlim_from = max_range
    xlim_to = -max_range

    ylim_from = max_range
    ylim_to = -max_range
    
    mySelector = gr.DiagramSelector(use=True,
                                    limit=np.inf,
                                    point_type="finite")

    for PI in data_PIs_betti:

        PI = mySelector.__call__(PI)
        
        if len(PI) > 0:
            x_from, y_from = np.amin(PI, axis=0)
            xlim_from = min(xlim_from, x_from)
            ylim_from = min(ylim_from, y_from)

            x_to, y_to = np.amax(PI, axis=0)
            xlim_to = max(xlim_to, x_to)
            ylim_to = max(ylim_to, y_to)

    # if ((xlim_from == max_range) or
    #     (xlim_to == -max_range) or
    #     (ylim_from == max_range) or
    #     (ylim_to == -max_range)):
        
    #     raise(ValueError("Unexpected PI range."))

    if ((xlim_from > xlim_to) or (ylim_from > ylim_to)):
        raise(ValueError("Error: min range greater than max."))

    return xlim_from, ylim_from, xlim_to, ylim_to


def write_PI_diagrams_for_multiple_fnl(data_PIs, data_labels, num_repeats=None, max_dim=None, write_folder="diagrams", num_fnls=None):
   """Overlay num_repeat PIs for different fnls, assuming data_PIs[betti] samples cycle through fnls consistently."""

   if num_fnls is None:
       num_fnls = len(set(data_labels[0]))
   if num_repeats is None:
       num_repeats = []
       for diag_list in data_PIs:
           num_repeats.append(
               len(diag_list)//num_fnls)

   if max_dim is None:
       max_dim = len(data_PIs)
       
   # hardcoded colors - red, blue, green - change it
   colour_list = ('red',
                  'blue',
                  'green',
                  'black',
                  'orange',
                  'cyan',
                  'purple',
                  'teal',
                  'magenta',
                  'pink'
                 )

   os.makedirs(write_folder, exist_ok=True)
   
   for betti_num in range(max_dim):
      plot.clf()
      
      data_PIs_betti = data_PIs[betti_num]
      data_labels_betti = data_labels[betti_num]

      if (len(data_PIs_betti) > 0) & (len(data_labels_betti) > 0):
          xlim_from, ylim_from, xlim_to, ylim_to = find_bounding_box_for_PI(data_PIs_betti)

          PIs_and_labels = list(zip(data_PIs_betti, data_labels_betti))

          for sample in range(num_repeats[betti_num]):
             for PI, label in PIs_and_labels[num_fnls*sample:num_fnls*(sample+1)]:

                # print(PI)
                # gd.plot_persistence_diagram(PD) # works on PD's only
                plot.scatter(PI[:,0], PI[:,1], color=colour_list[label])

             plot.title("PI Betti " + str(betti_num))
                # plot.show()
             if (xlim_from < xlim_to):
                 plot.xlim([xlim_from, xlim_to])
             plot.ylim([ylim_from, ylim_to])
             plot.savefig(write_folder + "/PI_Betti_" + str(betti_num) + "_sample_" +
                          str(sample) + '_red_0_blue_1')
             plot.clf()


#################################################
# ALM-BASED SAMPLE POINTS
#################################################
STRIDE_FACTOR_MORE_THAN_RES = 2
def make_grid(pixel,
              patch_width=16,
              max_angular_period=np.deg2rad(2),
              num_periods=2,
              nside=1024):

    patch_angle_width=max_angular_period*num_periods
    patch_stride=patch_angle_width/(patch_width-1)

    if patch_stride < STRIDE_FACTOR_MORE_THAN_RES*hp.nside2resol(nside): #to ensure that center pixel remains on a grid
      raise(ValueError("Patch stride too small for given nside."))

    working_patch = np.zeros((patch_width, patch_width), dtype=int)
    nx, ny = (patch_width, patch_width)

    x = np.linspace(0, patch_angle_width, nx)
    y = np.linspace(0, patch_angle_width, ny)
    
    perp_z = hp.pix2vec(nside, pixel)
    z_axis = np.array((0,0,1))
    alpha_to_z = calc_alpha_angle(z_axis, perp_z)
    perp_axis = np.cross(perp_z, z_axis)
    perp_axis = alpha_to_z*perp_axis/np.linalg.norm(perp_axis)
    r_perp = R.from_rotvec(perp_axis)

    perp_x = patch_stride*r_perp.apply((1, 0, 0))
    perp_y = patch_stride*r_perp.apply((0, 1, 0))
    r_x = R.from_rotvec(perp_x)
    r_y = R.from_rotvec(perp_y)

    grid_pixel_x = perp_z
    grid_pixel_x_y = grid_pixel_x
    # grid_vec = [] # np.zeros((patch_width, patch_width, 3))
    # rewrite with map function on mesh xv, yv
    for mi, mx in enumerate(x):
        for mj, my in enumerate(y):
            working_patch[mi, mj] = hp.vec2pix(nside, *grid_pixel_x_y)
            # # #grid_vec[mi, mj] = grid_pixel_x_y
            # grid_vec.append(grid_pixel_x_y)
            
            grid_pixel_x_y = r_y.apply(grid_pixel_x_y)

        grid_pixel_x = r_x.apply(grid_pixel_x)
        grid_pixel_x_y = grid_pixel_x

    # vec_row = grid_vec[0]-grid_vec[1]
    # vec_col = grid_vec[0]-grid_vec[patch_width]
    # np.testing.assert_almost_equal(calc_alpha_angle(vec_row, vec_col), np.pi/2, 6)

    # visualize_grid(working_patch[:,:])
    # visualize_point_set(np.array(grid_vec))
    
    return working_patch

def rescale_range_patch(patch):
    min_patch = np.amin(patch)
    max_patch = np.amax(patch)
    peak_to_peak = max_patch - min_patch
    if peak_to_peak > 0:
        patch[...] = (patch - min_patch)/peak_to_peak
#    return patch no need, in place

def calc_homology_dim(patch_width, alm=True, alm_fold=True):

    if alm:
        return num_points_upper_triangle(patch_width) if alm_fold else (patch_width**2)//2
    else:
        return patch_width**2

def num_points_upper_triangle(patch_width):
    num_points = (patch_width//2)
    return num_points*(num_points + 1)//2

def average_row_mag_freqs_leave_out_zero_and_nyquist(transformed_patch, patch_width):
    return (transformed_patch[1:patch_width//2,:] + \
            transformed_patch[patch_width//2:-1,:])/2

def extract_upper_triangle(average_across_diagonal, patch_width):
    return np.array(average_across_diagonal[
                np.triu_indices(patch_width//2)].tolist())

def kaiser_window_2D(M, beta=14):
    mid = (M-1)/2
    mid_diag = np.sqrt(2*(mid**2)) #new 'M'
    alpha = (mid_diag*2-1)/2.0
    
    window = np.zeros((M,M))
    max_val = 0
    
    for i in range(M):
        for j in range(M):
            n = np.sqrt((i-mid)**2 + (j-mid)**2) + mid_diag - 1
            window[i,j] = np.i0(beta * np.sqrt(1-((n-alpha)/alpha)**2.0))/np.i0(float(beta)) #from numpy
            max_val = max(max_val, window[i,j])

    window = window/max_val
    
    return window

def get_single_point(random_pixel_ids, sky_map,
                     patches_to_average=1,
                     threshhold=2,
                     patch_width=16, # needs to be power of 2
                     max_angular_period=np.deg2rad(1), #not of the window but of the patch
                     num_periods=2, #of the patch
                     above_threshhold=True,
                     norm_not_product=True, 
                     nside=512,
                     alm=True,
                     alm_fold=True,
                     alm_filter_to_apply=None,
                     mask=None
                     # tries to avoid edge effects, still to add window function
                     ):

    #threshhold=2 just hints that a good calibration would be in terms of standard deviations.

    """ Return a single alm-point, i.e. take a patch of temp sky, fourier transform and fold across same abs freq vectors. A point is returned if the norm/product of the fourier transform is 'greater than the raw threshhold' (if above_threshhold is true, otherwise 'less than the raw threshhold'). Absolute_temp chooses to take the absolute value of the product before comparison, otherwise not.
    """    
    attempt_id = 0
    num_pixel_ids = len(random_pixel_ids)
    patches_found = 0
    homology_dim = calc_homology_dim(patch_width, alm, alm_fold)
    make_homology_point = np.zeros(homology_dim)
    
    while (attempt_id < num_pixel_ids) and (patches_found < patches_to_average):

        grid = make_grid(random_pixel_ids[attempt_id], 
                         patch_width=patch_width,
                         max_angular_period=max_angular_period,
                         num_periods=num_periods,
                         nside=nside)
        
        if mask is None or np.all(mask[grid]==1):
            
            working_patch = sky_map[grid]
            if alm and alm_filter_to_apply is not None:
                working_patch = working_patch * alm_filter_to_apply

            if alm:
            # excluding last column (nyquist freq)
                transformed_patch = np.real(np.fft.rfft2(working_patch, norm="ortho")[:,:-1])
                if alm_fold:
                    transformed_patch_row_averaged = average_row_mag_freqs_leave_out_zero_and_nyquist(
                        transformed_patch, patch_width)

                    transformed_patch_row_averaged = np.vstack((transformed_patch[0,:],
                                                       transformed_patch_row_averaged))

                    average_across_diagonal = (transformed_patch_row_averaged +
                                               transformed_patch_row_averaged.T)/2
                    #could flip back before flattening but no real need to
                    homology_point = extract_upper_triangle(average_across_diagonal, patch_width)
                else:
                    homology_point = transformed_patch.flatten()


            else:
                homology_point = working_patch.flatten()

            if threshhold is not None:
                norm_or_product = np.linalg.norm(homology_point) if norm_not_product else \
                    np.abs(np.multiply.reduce(homology_point))

                if (norm_or_product < threshhold)^above_threshhold: #flip if above_threshhold true
                    #return homology_point, attempt_id
                    make_homology_point += homology_point
                    patches_found += 1
            else:
                make_homology_point += homology_point
                patches_found += 1

        attempt_id +=1
            

    if (attempt_id >= num_pixel_ids) and (patches_found < patches_to_average):
        print("Warning, did not find enough combination of pixels whose product is "
              "above/below the threshhold", flush=True)

        if (patches_found > 0):
            print("At least found some patches, using those.", flush=True)

    if (patches_found > 0):
        return make_homology_point/patches_found, attempt_id - 1
    else:   
        return None, None

def generate_sample_set(num_samples=100,
                        sample_multiplier=3,
                        patches_to_average=1,
                        fnl=0,
                        nside=512,
                        patch_width=32,
                        max_period_deg=2,
                        num_periods=3,
                        std_factor=1,
                        above_threshhold=True,
                        norm_not_product=True,
                        #random_pixel_ids=None,
                        alm=True,
                        alm_fold=True,
                        alm_do_filter=True,
                        rescale_range=True,
                        sky_map=None,
                        mask=None, **ignore_others): # slightly dangerous

            if alm and alm_do_filter:
                alm_filter_to_apply = kaiser_window_2D(patch_width)
            else:
                alm_filter_to_apply = None
                
            if sky_map is None:
                scaled_sky_map = make_map_from_alm(nside,
                                    configcmb.full_path_sim_map,
                                    configcmb.full_path_sim_map_ng,
                                    fnl=fnl)
            else:
                scaled_sky_map = sky_map

            #if random_pixel_ids is None:
            random_pixel_ids = gen_random_pixel_ids(
                    num_samples*sample_multiplier*patches_to_average, nside)
                
            num_pixel_ids = len(random_pixel_ids) 

            num_dims = calc_homology_dim(patch_width, alm, alm_fold)

            if std_factor is None:
                threshhold = None
            else:
                sky_std = np.sqrt(np.var(scaled_sky_map))
                threshhold = sky_std*std_factor*num_dims \
                    if norm_not_product else (sky_std*std_factor/5.5)**(num_dims)

            homology_points = []
            # found_pixel_ids = []
            # homology_type = np.dtype([('homology_point'+str(num_dims), np.float32,
            #                             (num_dims,))])
            last_index = 0
            num_found = 0
            skipped = 0
            while (num_found < num_samples) and (last_index < num_pixel_ids):
                homology_point, new_relative_index =\
                    get_single_point(random_pixel_ids[last_index:],
                                     patches_to_average=patches_to_average,
                                     sky_map=scaled_sky_map,
                                     threshhold=threshhold,
                                     patch_width=patch_width,
                                     max_angular_period=np.deg2rad(max_period_deg),
                                     num_periods=num_periods,
                                     above_threshhold=above_threshhold,
                                     norm_not_product=norm_not_product,
                                     nside=nside,
                                     alm=alm,
                                     alm_fold=alm_fold,
                                     alm_filter_to_apply=alm_filter_to_apply,
                                     mask=mask)
                
                if new_relative_index is not None:
                    skipped += max(0, new_relative_index - patches_to_average + 1)
                    last_index = last_index + new_relative_index
                    # found_pixel_ids.append(random_pixel_ids[last_index])
                    num_found += 1
                    last_index += 1
                    #homology_points.append(np.array(tuple([homology_point]), dtype=homology_type))
                    homology_points.append(homology_point)
                else:
                    skipped += num_pixel_ids - last_index
                    print("Warning: not enough candidates meet the threshhold. Continuing with:",
                          num_found, " points.")
                    break

            if rescale_range:
                for patch in homology_points:
                        rescale_range_patch(patch)

            return homology_points, skipped #found_pixel_ids, skipped # number of points passed over while looking for points that meet the threshhold

        
# ###################### HOMOLOGY ####################### #

@ray.remote(max_retries=-1)
def calc_rips_complex(points, rips_dim=configcmb.max_dim):

    rips_complex = gd.RipsComplex(points=points)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(rips_dim)

    rips_PD = simplex_tree.persistence()
    rips_PI = []
    for j in range(rips_dim):
        rips_PI.append(simplex_tree.persistence_intervals_in_dimension(j))

    return rips_PD, rips_PI

def calc_rips_complex_for_multiple_fnl(sample_fnls,
                                       fnls=configcmb.sim_map_fnl_set,
                                       rips_dim=configcmb.max_dim,
                                       from_snapshot=None,
                                       to_snapshot=None):
    """
    sample_fnls is a list of repeats of a list of fnls or a list of homology points of a list of co-ords.
    """

    # from_snapshot provided and sample_fnls is empty
    if (from_snapshot!=None):
      sample_fnls = pickle.load(open(from_snapshot, 'rb'))

    data_PIs = [[] for _ in range(rips_dim)]
    data_labels = [[] for _ in range(rips_dim)]
    repeat_fnls = len(sample_fnls)

    ptasks_repeat = []

    print()

    # Measure Ray parallel execution
    start_time = datetime.now()

    for i in tqdm(range(repeat_fnls), "CMB Homology - Ray Parallel"):

       ptasks = []

       for j in range(len(fnls)):

          # ray parallel call
          ptasks.append(calc_rips_complex.remote(sample_fnls[i][j],
                                                 rips_dim=rips_dim))

       ptasks_repeat.append(ptasks)

    for ptasks in ptasks_repeat:
       rips_PD_PIs = ray.get(ptasks)

       for j, j_fnl in enumerate(fnls):
          rips_PI = rips_PD_PIs[j][1]
	  
          for k in range(rips_dim):

    	    # Ray workaround - copy to make read-write again
            data_PIs[k].append(np.copy(rips_PI[k]))
            data_labels[k].append(j_fnl)
            
            if len(rips_PI[k]) == 0:
                print("Warning, some diagrams were empty and therefore excluded. fnl: ", j_fnl, ". Betti: ", k, flush=True)

    time_elapsed = datetime.now() - start_time

    print('Elapsed (hh:mm:ss:ms) {}'.format(time_elapsed))

    # save homology for reuse
    if to_snapshot!=None:

      print("Saving CMB Homology")

      with open(to_snapshot, 'wb') as f:
        pickle.dump((data_PIs, data_labels), f)

    return data_PIs, data_labels


# ###################### ML PIPELINE ####################### #


def ML_train_test(data_PIs,
                  data_labels,
                  rips_dim=configcmb.max_dim, \
                  test_size=0.4, \
                  cv=4, \
                  from_snapshot=None
                 ):

   if (from_snapshot!=None):
     data_PIs, data_labels = pickle.load(open(from_snapshot, 'rb'))  
    
   print("Finished Sampling and Homology. Starting ML Training.", flush=True)

   # TRAIN and TEST SPLIT

   for betti in tqdm(range(rips_dim), "ML Training..."):

      tic = time.perf_counter()

      print("Executing ML for betti: ", betti, flush=True)

      if len(data_PIs[betti]) > 2:#configcmb.min_bin_points:
        perm                 = np.random.permutation(len(data_labels[betti]))
        limit                = np.int32(test_size * len(data_labels[betti]))
        test_sub, train_sub  = perm[:limit], perm[limit:]
        train_labels         = np.array(data_labels[betti])[train_sub]
        test_labels          = np.array(data_labels[betti])[test_sub]
        train_PIs            = [data_PIs[betti][i] for i in train_sub]
        test_PIs             = [data_PIs[betti][i] for i in test_sub]

        # DECLARE DEFAULT PIPELINE

        # ("Padder", gr.Padding(use=True)),
        # (slice(None,None),slice(0,1))
        # ("PadIgnorePadFlag", ColumnTransformer([("pad", gr.Padding(use=True),
        #                                   ])),
        # ("MyPadding", MyPadding(use=True))
        pipe = Pipeline([
             ("Separator", gr.DiagramSelector(use=True, limit=np.inf,
                                              point_type="finite")),
             ("MyPadding", MyPadding(use=True)),
             ("Scaler",    gr.DiagramScaler()),
             ("TDA",       gr.PersistenceImage()),
             ("Estimator", SVC())])

        # VARIANTS OF DEFAULT PIPELINE

        #    "TDA__weight":         [lambda x: np.arctan(x[1]-x[0])],
        #    "TDA__sample_range":  [[0, 1]],

        param_scalers = [[([0,1], MinMaxScaler())],
                         [([0,1], Normalizer())],
                         [([0,1], RobustScaler())]
                        ]
        param_tda_band = [0.01, 0.1, 0.4]

        param =    [{"Scaler__use":        [True],
                     "MyPadding__use":     [True, False],
                     "Scaler__scalers":    param_scalers,
                     "TDA":                [gr.SlicedWassersteinKernel()],
                     "TDA__bandwidth":     param_tda_band,
                     "TDA__num_directions":[20, 40],
                     "Estimator":          [SVC(kernel="precomputed", gamma="auto")]},

                    {"Scaler__use":        [False],
                     "MyPadding__use":     [True, False],
                     "TDA":                [gr.SlicedWassersteinKernel()],
                     "TDA__bandwidth":     param_tda_band,
                     "TDA__num_directions":[20, 40],
                     "Estimator":          [SVC(kernel="precomputed", gamma="auto")]},

                    {"Scaler__use":         [True],
                     "MyPadding__use":      [True, False],
                     "Scaler__scalers":     param_scalers,
                     "TDA":                 [gr.PersistenceWeightedGaussianKernel()],
                     "TDA__bandwidth":      param_tda_band,
                     "Estimator":           [SVC(kernel="precomputed", gamma="auto")]},

                    {"Scaler__use":        [False],
                     "MyPadding__use":     [True, False],
                     "TDA":                [gr.PersistenceWeightedGaussianKernel()],
                     "TDA__bandwidth":     param_tda_band,
                     "Estimator":          [SVC(kernel="precomputed", gamma="auto")]},

                    {"Scaler__use":         [True],
                     "Scaler__scalers":     param_scalers,
                     "TDA":                 [gr.PersistenceImage()],
                     "TDA__bandwidth":      param_tda_band,
                     "TDA__resolution":     [ [5,5], [20,20], [40, 40] ],
                     "Estimator":           [SVC()]},

                    {"Scaler__use":        [False],
                     "TDA":                [gr.PersistenceImage()],
                     "TDA__resolution":    [ [5,5], [20,20], [40, 40] ],
                     "TDA__bandwidth":     param_tda_band,
                     "Estimator":          [SVC()]},

                    {"Scaler__use":        [False],
                     "MyPadding__use":     [True, False],
                     "TDA":                [gr.PersistenceImage()],
                     "TDA__resolution":    [ [5,5], [20,20], [40, 40] ],
                     "Estimator":          [RandomForestClassifier()]},

                    {"Scaler__use":        [True],
                     "Scaler__scalers":    param_scalers,
                     "MyPadding__use":     [False],
                     "TDA":                [gr.Landscape()],
                     "TDA__resolution":    [50, 100],
                     "Estimator":          [RandomForestClassifier()]},

                    {"Scaler__use":        [False],
                     "MyPadding__use":     [False],
                     "TDA":                [gr.Landscape()],
                     "TDA__resolution":    [50, 100],
                     "Estimator":          [RandomForestClassifier()]},

                    {"Scaler__use":         [True],
                     "Scaler__scalers":     param_scalers,
                     "MyPadding__use":      [True, False],
                     "TDA":                [gr.BottleneckDistance()],
                     "TDA__epsilon":       param_tda_band,
                     "Estimator":          [KNeighborsClassifier(metric="precomputed")]}
                    ]

        # DEFINE MODEL
        # model = GridSearchCV(pipe, param, scoring=roc_auc_score, cv=cv, n_jobs=-1) # not working yet
        model = GridSearchCV(pipe, param, cv=cv, n_jobs=-1)
        
        # TRAIN
        model = model.fit(train_PIs, train_labels)

        print(model.best_params_, flush=True)

        print("Betti " + str(betti) + " Train accuracy = " + str(model.score(train_PIs, train_labels)), flush=True)
        print("Betti " + str(betti) + " Test accuracy  = " + str(model.score(test_PIs,  test_labels)), flush=True)

        print("Grid scores on Training Set : ", flush=True)
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
           print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), flush=True)

        toc = time.perf_counter()
        print(f"Train_and_Test in {toc - tic:0.4f} seconds", flush=True)


#################################################
# STATISTICAL TOOLS
#################################################


def permutation_test(data_PIs, data_labels, test_betti=2,
                     num_repeats=10,
                     num_perms=30,
                     distmetric="pot_wasserstein",
                     internal_p=2,
                     n_jobs=4,
                     visualise=None):
   #                  to_snapshot=None): leave out until have a corresponding from_snapshot
   
   """
   Perform permuation test. data_PIs are a list (betti numbers) of a list (samples) of diagrams.
   For now assume data_labels are either 0 or 1. Built in distmetric ignores points close to the diagonal.
   """
   # distmetric = "hera_wasserstein"
   # distmetric = "bottleneck"
   # distmetric = 'persistence_fisher'
   # order=1

   fnl_0_index = 0
   # fnl_1_index = len(set(data_labels[test_betti]))-1 # doesn't work if all PD's left out!
   fnl_1_index = 1 # just stick to label convention
   
   fnl0s = [data_PIs[test_betti][int(i)] for i in
           np.where(np.array(data_labels[test_betti]) == fnl_0_index)[0]]


   fnl1s = [data_PIs[test_betti][int(i)] for i in
             np.where(np.array(data_labels[test_betti]) == fnl_1_index)[0]]

   min_num = min(len(fnl0s), len(fnl1s), num_repeats) # num_mixed_list//2
   mixed_list = fnl0s[0:min_num] + fnl1s[0:min_num]

   my_kwargs = {'internal_p':internal_p, 'metric':distmetric,
                 'n_jobs':n_jobs}

   all_t_vals = []

   num_mixed_list = len(mixed_list)
   
   # print()

   for permutation in tqdm(range(num_perms),
                            "Permutation Test - Betti " + str(test_betti)):
      if permutation==0:
         permute_indices = range(num_mixed_list)
      else:
         permute_indices = \
            np.random.permutation(num_mixed_list)

      set0 = [mixed_list[i] for i in \
              permute_indices[:min_num]] #num_mixed_list//2

      set1 = [mixed_list[i] for i in \
              permute_indices[min_num:]]

      mydistances_0_to_1 = \
         gr.metrics.pairwise_persistence_diagram_distances(
            set0,set1, **my_kwargs)

      mydistances_0_to_0 = \
         gr.metrics.pairwise_persistence_diagram_distances(
            set0,set0, **my_kwargs)

      mydistances_1_to_1 = \
         gr.metrics.pairwise_persistence_diagram_distances(
            set1,set1, **my_kwargs)

      num_set0 = len(set0)
      each0 = np.zeros((num_set0, 2))
      each0[:,0] = np.sum(mydistances_0_to_0,
                              axis=0)/(num_set0)
      each0[:,1] = np.mean(mydistances_0_to_1, axis=1)

      num_set1 = len(set1)
      each1 = np.zeros((num_set1, 2))
      each1[:,0] = np.sum(mydistances_1_to_1,
                              axis=1)/(num_set1)
      each1[:,1] = np.mean(mydistances_0_to_1, axis=0)

      diff0 = np.maximum(each0[:,1] - each0[:,0], 0)
      averagediff0 = np.mean(diff0)
      
      diff1 = np.maximum(each1[:,1] - each1[:,0],0)
      averagediff1 = np.mean(diff1)

      # t_val = np.abs((averagediff1+averagediff0)/2) #for non-different sets, this should be close to 
      # zero, a negative average difference is fine and helps to bring the sum down
      # t_val = (np.abs(averagediff1)+np.abs(averagediff0))/2 #for non-different sets, this should be close to
      t_val =  (averagediff1+averagediff0)/2 # I prefer this formulation, which treats the average differences from different sets on an equal footing.
      all_t_vals.append(t_val)

      # Take a set_0 PI, measure it's average wasserstein to all other set_0 PI's and store that as y_d0. Then measure it's average wasserstein distance to all set_1 PI's, store that as y_d1. Plot: (0,y_d0) and (1,y_d1) and join them. Draw such a line for each set_0 in red and redo for each set_1 in blue.

      if visualise is not None:
         if permutation in visualise:
            plot.plot([0, 1], each0.transpose(), color='red')
            plot.plot([1, 0], each1.transpose(), color='blue')
            plot.show()
             # plot.clf()
            plot.close()

      p_val = np.sum(all_t_vals >= all_t_vals[0])/num_perms # 0 is our non-permutated base-case
      
   # # Save Permutation Test results
   # if (to_snapshot!=None):

   #    print("Saving CMB Permutation Test - Betti ", str(test_betti))

   #    with open(to_snapshot,'wb') as f:
   #     pickle.dump((mydistances_0_to_1, mydistances_0_to_0, 
   #                  mydistances_1_to_1), f)
 
   print("Final p-value for betti ", test_betti,": ", p_val, "\n")

   return(p_val)

############ BAYESIAN ######################

def posteriors_of_training(T, prior, unant, lik_cm):
    posts = [Posterior(DYO = diag, prior = prior, clutter = unant, sigma_DYO = lik_cm)
             for diag in tqdm(T, "Bayesian Training")]
    return posts

@ray.remote(max_retries=-1)
def prob_of_testing(D, posteriors_of_training):
    pp_log = 0
    for feat in D:
        post_pri = [p.eval(feat) for p in posteriors_of_training]
        pp_log += np.log10(np.sum(np.array(post_pri))+EPS)  #sum because prob of coming from train1 OR train 2, but normalization is missing, but not needed because cancel out in bayes factor
    return pp_log

def tilt_PIs(data_PIs):
    for betti in range(len(data_PIs)):
        for PI in data_PIs[betti]:
            PI[:,1] = PI[:,1] - PI[:,0] # death - birth


# TODO: consider writing out training/testing data

def visualize_posterior(xlim_from, ylim_from, xlim_to, ylim_to, posterior, write_folder=None, filename="trained_posterior.png"):
    # create grid for computing posterior intensity
    b = np.linspace(xlim_from, xlim_to, 20)
    p = np.linspace(ylim_from, ylim_to, 20)
    B,P = np.meshgrid(b,p)

    # evaluate posterior intensity over grid
    I = list()
    for ind in range(len(P)):
        l = list()
        for i in range(len(P)):
            l.append(posterior.eval([B[ind][i],P[ind][i]]))
        I.append(l)
    I = I / np.max(I)

    # plot posterior intensity
    plot.figure()
    plot.style.use('seaborn-bright')
    plot.contourf(B,P,I, 20, cmap = 'coolwarm', vmin = 0, vmax = 1)
    plot.title('Posterior Intensity')
    cb = plot.colorbar()
    cb.ax.set_title('Scal. Intens.')
    plot.gca().set_ylabel('Persistence')
    plot.gca().set_xlabel('Birth')
    plot.xlim([xlim_from, xlim_to])
    plot.ylim([ylim_from, ylim_to])
    #plot.scatter(pd_t[:,0],pd_t[:,1], s = 20 ,color = 'black') # overlay persistence diagram
    if write_folder is None:
        plot.show()
    else:
        plot.savefig(write_folder + "/" + filename)
    plot.close()
 

def _show_bayes_factor_distribution(scores, write_folder=None, filename="bayes.png"):
    # examine score distributions
    scores_0 = scores[0]
    scores_1 = scores[1]
    
    scores_0 = scores_0[:, 0] - scores_0[:, 1]
    scores_1 = scores_1[:, 0] - scores_1[:, 1]
    
    plot.hist(scores_0, label = '0', color = 'blue', alpha = 0.5)
    plot.hist(scores_1, label = '1', color = 'red', alpha = 0.5)
    plot.xlabel('$log p(0) / p(1) $')
    plot.ylabel('Count')
    plot.title('Bayes Factor Distributions')
    plot.legend()
    # plot.show()
    # plot.close()
    if write_folder is None:
        plot.show()
    else:
        plot.savefig(write_folder + "/" + filename)
    
    plot.clf()
    
    # compute aucs
    y0 = np.zeros(len(scores_0))
    y1 = np.ones(len(scores_1))
    
    y_true = np.concatenate([y0, y1])
    y_score = np.concatenate([scores_0, scores_1])
 
    tpr, fpr, _ = roc_curve(y_true, y_score)
    AUC = auc(fpr,tpr)
    
    return AUC
    
def bayesian_classification(data_PIs,
                            data_labels,
                            test_betti,
                            string_filename_suffix="",
                            write_folder="diagrams",
                            visualise=None):
                            # to_snapshot=None): leave out until have a corresponding from_snapshot

    """
    Perform Bayesian Training (75% of data) and Testing (25%) on two labelled tilted data_PI sets (0 and 1). Returning AUC on the testing data.
    """
   
    xlim_from, ylim_from, xlim_to, ylim_to = find_bounding_box_for_PI(data_PIs[test_betti])
    #assert(ylim_from < EPS)

    mid_x = (xlim_to + xlim_from)/2
    mid_y = (ylim_to + ylim_from)/2
    width_x = xlim_to - xlim_from
    width_y = ylim_to - ylim_from

    mu_pri = np.array([[mid_x, mid_y]])  # prior mean
    w_pri = np.array([1])                # prior weight
    sig_pri = np.array([max(width_x, width_y)])        # prior covariance magnitude

    PRIOR_PROP = 0.75
    SIGMA_DYO = min(width_x, width_y)/100 # 0.1
    
    # fnl_0_index = 0
    # fnl_1_index = 1 #len(set(data_labels[test_betti]))-1

    # fnl_0s_diags = [data_PIs[test_betti][int(i)] for i in
    #        np.where(np.array(data_labels[test_betti]) == fnl_0_index)[0]]

    # fnl_1s_diags = [data_PIs[test_betti][int(i)] for i in
    #          np.where(np.array(data_labels[test_betti]) == fnl_1_index)[0]]

    # min_num = min(len(fnl_0s_diags), len(fnl_1s_diags))

    # if min_num < 4:
    #     print("Not enough diagrams to do Bayesian train and test, skipping, betti ", test_betti, ".")
    #     return None

    # if visualise is None:
    #     os.makedirs(write_folder, exist_ok=True)
    #     for i, posts in tqdm(enumerate(posts_of_training_0)):
    #         visualize_posterior(xlim_from, ylim_from, xlim_to, ylim_to, posts, write_folder, "posterior_0_" + "betti_" + str(test_betti) + "_repeat_" + str(i) +
    #                             string_filename_suffix + ".png")
    #     for i, posts in tqdm(enumerate(posts_of_training_1)):
    #         visualize_posterior(xlim_from, ylim_from, xlim_to, ylim_to, posts, write_folder, "posterior_1_" + "betti_" + str(test_betti) + "_repeat_" + str(i) +
    #                             string_filename_suffix + ".png")
    # else:
    #     if test_betti in visualise:
    #         plot.show()    

    start_time = datetime.now()

    # # load data
    # DATA_PATH = '/home/chris/projects/bayes_tda/data/'
    # DATA = 'bccfcc_small.npy'
    # LABELS = 'bccfcc_small_labels.npy'

    # data = np.load(DATA_PATH + DATA, allow_pickle = True)
    # labels = np.load(DATA_PATH + LABELS, allow_pickle = True)

    # build prior and clutter
    prior = RGaussianMixture(mus = mu_pri, 
                             sigmas = sig_pri, 
                             weights = w_pri, 
                             normalize_weights= False)

    clutter = RGaussianMixture(mus = mu_pri, 
                               sigmas = sig_pri, 
                               weights = [[0]], # switch off via weights, better if can take None 
                               normalize_weights= False)

    classifier = EBFC(data = data_PIs[test_betti],
                      labels = data_labels[test_betti],
                      data_type = 'diagrams')

    scores = classifier.compute_scores(clutter, 
                                       prior,
                                       prior_prop = PRIOR_PROP,
                                       sigma_DYO = SIGMA_DYO)
    
    time_elapsed = datetime.now() - start_time 

    print('Elapsed (hh:mm:ss:ms) {}'.format(time_elapsed))

    # # Subtract fnl_0 Lists
    # zip_fnl_0s = zip(fnl_0_list0, fnl_0_list1)
    # log_prob_diff_testing_fnl_0s = []

    # for fnl_0_list0, fnl_0_list1 in zip_fnl_0s:

    #     log_prob_diff_testing_fnl_0s.append(fnl_0_list0 - fnl_0_list1)

    # # Subtract fnl_1 Lists
    # zip_fnl_1s = zip(fnl_1_list0, fnl_1_list1)
    # log_prob_diff_testing_fnl_1s = []

    # for fnl_1_list0, fnl_1_list1 in zip_fnl_1s:

    #     log_prob_diff_testing_fnl_1s.append(fnl_1_list0 - fnl_1_list1)

    # # save results
    # # if (to_snapshot!=None):

    # #    # print("Saving CMB Bayesian Test - Betti - ", str(test_betti))

    # #    with open(to_snapshot,'wb') as f:
    # #     pickle.dump((log_prob_diff_testing_fnl_0s,
    # #                   log_prob_diff_testing_fnl_1s), f)

    # my_bins = np.histogram_bin_edges([log_prob_diff_testing_fnl_0s, 
    #                                   log_prob_diff_testing_fnl_1s], bins=50)

    # plot.hist(log_prob_diff_testing_fnl_0s, bins=my_bins, alpha = 0.3)
    # plot.hist(log_prob_diff_testing_fnl_1s, bins=my_bins, alpha = 0.3)
    # plot.legend(['0','1'])
    # plot.xlabel('Log Bayes Factor')
    # plot.ylabel('Number of Validation Examples')

    # examine score distributions

    auc = _show_bayes_factor_distribution(scores, filename=string_filename_suffix + "_" + str(test_betti), write_folder=write_folder)
    print('AUC: ' + str(auc))

    # if visualise is None:
    #     os.makedirs(write_folder, exist_ok=True)
    #     plot.savefig(write_folder + "/Bayesian_Classification_betti_" + str(test_betti) +
    #                  string_filename_suffix)
    # else:
    #     if test_betti in visualise:
    #         plot.show()

    # plot.clf()

    # num_testing_0 = len(log_prob_diff_testing_fnl_0s)
    # num_testing_1 = len(log_prob_diff_testing_fnl_1s)

    # class_labels = [1]*num_testing_1 + [0]*num_testing_0 # list concatenated, 0 on right, in keeping with increasing log prob

    # log_prob_diff_testing  = log_prob_diff_testing_fnl_1s + log_prob_diff_testing_fnl_0s

    # fpr, tpr, thresholds  = roc(class_labels, log_prob_diff_testing, pos_label=0) #roc expecting pred to be increasing which we do satisfy, roc's default class label on the right is 1, which we need to swap

    # # plot.plot(fpr, tpr)
    # # plot.show()

    # auc_value = auc(fpr, tpr)
    # print("AUC: ", auc_value, "\n")

    return auc
    

# def bayesian_classification(data_PIs,
#                             data_labels,
#                             test_betti,
#                             number_to_classify = None,
#                             string_filename_suffix="",
#                             write_folder="diagrams",
#                             visualise=None):
#                             # to_snapshot=None): leave out until have a corresponding from_snapshot

#     """
#     Perform Bayesian Training (75% of data) and Testing (25%) on two labelled tilted data_PI sets (0 and 1). Returning AUC on the testing data.
#     If num_to_classify is not None, then Train on 100% of the data except the last num_to_classify. Classify the last num_to_classify. Returning label and Log Bayes factor. Else return AUC.
#     """
   
#     xlim_from, ylim_from, xlim_to, ylim_to = find_bounding_box_for_PI(data_PIs[test_betti])
#     assert(ylim_from < EPS)
#     mid_x = (xlim_to - xlim_from)/4
#     mid_y = (ylim_to - ylim_from)/2

#     mu_pri = np.array([[xlim_from, ylim_from],
#                        [xlim_from+mid_x, ylim_from+mid_y],
#                        [xlim_from + 2*mid_x, ylim_to],
#                        [xlim_to-mid_x, ylim_from+mid_y],
#                        [xlim_to, ylim_from]])  # prior mean
#     w_pri = np.array([1, 1, 1, 1, 1]) # prior weight
#     sig_pri = np.array([mid_x/10, mid_x/10, mid_x/10,
#                         mid_x/10, mid_x/10]) # prior covariance magnitude
#     pri_train = Prior(weights=w_pri, mus=mu_pri, sigmas=sig_pri)
#     # unanticipated intensity set to zero
#     unant = Prior(weights = np.array([0, 0, 0, 0, 0]), mus=mu_pri, sigmas=sig_pri)

#     fnl_0_index = 0
#     fnl_1_index = 1 #len(set(data_labels[test_betti]))-1
      
#     fnl_0s_diags = [data_PIs[test_betti][int(i)] for i in
#            np.where(np.array(data_labels[test_betti]) == fnl_0_index)[0]]

#     fnl_1s_diags = [data_PIs[test_betti][int(i)] for i in
#              np.where(np.array(data_labels[test_betti]) == fnl_1_index)[0]]
    
#     min_num = min(len(fnl_0s_diags), len(fnl_1s_diags))

#     if min_num < 4:
#         print("Not enough diagrams to do Bayesian train and test, skipping, betti ", test_betti, ".")
#         return None

#     if number_to_classify is None:
#         num_train = int(min_num*0.75)
#     else:
#         num_train = min_num

#     fnl_0_diags_train = fnl_0s_diags[0:num_train]
#     fnl_1_diags_train = fnl_1s_diags[0:num_train]

#     fnl_0_diags_testing = fnl_0s_diags[num_train:]
#     fnl_1_diags_testing = fnl_1s_diags[num_train:]

#     # TRAINING
    
#     posts_of_training_0 = posteriors_of_training(T=fnl_0_diags_train, prior = pri_train,
#                                                  unant = unant,lik_cm = 0.1)
#     posts_of_training_1 = posteriors_of_training(T=fnl_1_diags_train, prior = pri_train,
#                                                  unant = unant,lik_cm = 0.1)
#     # END OF TRAINING

#     if visualise is None:
#         os.makedirs(write_folder, exist_ok=True)
#         for i, posts in tqdm(enumerate(posts_of_training_0)):
#             visualize_posterior(xlim_from, ylim_from, xlim_to, ylim_to, posts, write_folder, "posterior_0_" + "betti_" + str(test_betti) + "_repeat_" + str(i) +
#                                 string_filename_suffix + ".png")
#         for i, posts in tqdm(enumerate(posts_of_training_1)):
#             visualize_posterior(xlim_from, ylim_from, xlim_to, ylim_to, posts, write_folder, "posterior_1_" + "betti_" + str(test_betti) + "_repeat_" + str(i) +
#                                 string_filename_suffix + ".png")
#     else:
#         if test_betti in visualise:
#             plot.show()
    
#     if number_to_classify is not None:

#         #if data_labels[-1] is not None:
            
#         diag_to_classify = data_PIs[test_betti][-number_to_classify:]
        
#         log_prob_0_handle = [prob_of_testing.remote(diag_sky, posts_of_training_0)
#                                     for diag_sky in tqdm(diag_to_classify, "Bayesian Classification Sky vs Training 0- Ray Parallel")]
#         log_prob_1_handle = [prob_of_testing.remote(diag_sky, posts_of_training_1)
#                                     for diag_sky in tqdm(diag_to_classify, "Bayesian Classification Sky vs Training 1- Ray Parallel")]

#         log_prob_0 = np.array(ray.get(log_prob_0_handle))
#         log_prob_1 = np.array(ray.get(log_prob_1_handle))

#         log_bayes_factors = log_prob_0 - log_prob_1

#         vote_for_0 = np.sum(log_bayes_factors > 0)/(number_to_classify)
#         diag_label = 0 if vote_for_0 > 0.5   else 1
#         vote = vote_for_0 if diag_label==0 else 1 - vote_for_0

#         return diag_label, vote, log_bayes_factors
    
#     else:
#         # TESTING (log prob of diag coming from 0 MINUS log prob of diag coming from 1, so if actually came from 0 this should be small negative - large negative therefore should be positive, and if came from 1 should be large negative - small negative, therefore large negative. So testing 0 should be right of testing 1.

#         # Measure Ray parallel execution
#         start_time = datetime.now()

#         # Parallelize - Ray
#         testing_diag0_training_0 = [prob_of_testing.remote(diag0, posts_of_training_0)
#                                     for diag0 in tqdm(fnl_0_diags_testing, "Bayesian Testing fnl_0 0 - Ray Parallel")]

#         testing_diag0_training_1 = [prob_of_testing.remote(diag0, posts_of_training_1)
#                                     for diag0 in tqdm(fnl_0_diags_testing, "Bayesian Testing fnl_0 1 - Ray Parallel")]

#         testing_diag1_training_0 = [prob_of_testing.remote(diag1, posts_of_training_0)
#                                     for diag1 in tqdm(fnl_1_diags_testing, "Bayesian Testing fnl_1 0 - Ray Parallel")]

#         testing_diag1_training_1 = [prob_of_testing.remote(diag1, posts_of_training_1)
#                                     for diag1 in tqdm(fnl_1_diags_testing, "Bayesian Testing fnl_1 1 - Ray Parallel")]

#         fnl_0_list0 = ray.get(testing_diag0_training_0)
#         fnl_0_list1 = ray.get(testing_diag0_training_1)

#         fnl_1_list0 = ray.get(testing_diag1_training_0)
#         fnl_1_list1 = ray.get(testing_diag1_training_1)

#         time_elapsed = datetime.now() - start_time 

#         print('Elapsed (hh:mm:ss:ms) {}'.format(time_elapsed))

#         # Subtract fnl_0 Lists
#         zip_fnl_0s = zip(fnl_0_list0, fnl_0_list1)
#         log_prob_diff_testing_fnl_0s = []

#         for fnl_0_list0, fnl_0_list1 in zip_fnl_0s:

#             log_prob_diff_testing_fnl_0s.append(fnl_0_list0 - fnl_0_list1)

#         # Subtract fnl_1 Lists
#         zip_fnl_1s = zip(fnl_1_list0, fnl_1_list1)
#         log_prob_diff_testing_fnl_1s = []

#         for fnl_1_list0, fnl_1_list1 in zip_fnl_1s:

#             log_prob_diff_testing_fnl_1s.append(fnl_1_list0 - fnl_1_list1)

#         # save results
#         # if (to_snapshot!=None):

#         #    # print("Saving CMB Bayesian Test - Betti - ", str(test_betti))

#         #    with open(to_snapshot,'wb') as f:
#         #     pickle.dump((log_prob_diff_testing_fnl_0s,
#         #                   log_prob_diff_testing_fnl_1s), f)

#         my_bins = np.histogram_bin_edges([log_prob_diff_testing_fnl_0s, 
#                                           log_prob_diff_testing_fnl_1s], bins=50)

#         plot.hist(log_prob_diff_testing_fnl_0s, bins=my_bins, alpha = 0.3)
#         plot.hist(log_prob_diff_testing_fnl_1s, bins=my_bins, alpha = 0.3)
#         plot.legend(['0','1'])
#         plot.xlabel('Log Bayes Factor')
#         plot.ylabel('Number of Validation Examples')

#         if visualise is None:
#             os.makedirs(write_folder, exist_ok=True)
#             plot.savefig(write_folder + "/Bayesian_Classification_betti_" + str(test_betti) +
#                          string_filename_suffix)
#         else:
#             if test_betti in visualise:
#                 plot.show()

#         plot.clf()

#         num_testing_0 = len(log_prob_diff_testing_fnl_0s)
#         num_testing_1 = len(log_prob_diff_testing_fnl_1s)

#         class_labels = [1]*num_testing_1 + [0]*num_testing_0 # list concatenated, 0 on right, in keeping with increasing log prob

#         log_prob_diff_testing  = log_prob_diff_testing_fnl_1s + log_prob_diff_testing_fnl_0s

#         fpr, tpr, thresholds  = roc(class_labels, log_prob_diff_testing, pos_label=0) #roc expecting pred to be increasing which we do satisfy, roc's default class label on the right is 1, which we need to swap

#         # plot.plot(fpr, tpr)
#         # plot.show()

#         auc_value = auc(fpr, tpr)
#         print("AUC: ", auc_value, "\n")

#         return auc_value

# ################### MAIN ################### #
if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(num_cpus=4)

    fnl_0 = 0
    num_repeats = 20
    rips_dim_post_process_start = 1
    rips_dim_max = 3

    do_sim = True # simulated sky (either way do rips): permutation test and bayesian
    do_real_sky = False
    if do_real_sky:
        num_repeats_sky = 10
    else:
        num_repeats_sky = 0

    output_list = [] #inner_most list: fnl_1, num_samples, (betti, p), (betti, AUC), (betti, label, vote, bayes_factors)
    #    nested_levels = 14

    for num_samples in [100]: # range(100, 1000, 100):
        inner_list1 = output_list
        inner_list1.append([])
        inner_list2 = inner_list1[-1]
        for fnl_1 in [100]: # range(300, 1000, 300):#[1000]:
            inner_list2.append([])
            inner_list3 = inner_list2[-1]
            for nside in [2048]:
             scaled_sky_map_fnl_0 = make_map_from_alm(nside,
                                                     configcmb.full_path_sim_map,
                                                     None,
                                                     fnl=None)
             scaled_sky_map_fnl_1 = make_map_from_alm(nside,
                                               configcmb.full_path_sim_map,
                                               configcmb.full_path_sim_map_ng,
                                               fnl=fnl_1)
             inner_list3.append([])
             inner_list4 = inner_list3[-1]
             for sample_multiplier in [1]:
              inner_list4.append([])
              inner_list5 = inner_list4[-1]
              for patches_to_average in [1]:
               inner_list5.append([])
               inner_list6 = inner_list5[-1]
               for patch_width in [16]:
                inner_list6.append([])
                inner_list7 = inner_list6[-1]
                for max_period_deg in [1]:
                 inner_list7.append([])
                 inner_list8 = inner_list7[-1]
                 for num_periods in [2]:
                  constraint = np.deg2rad(max_period_deg)*num_periods/(patch_width-1)
                  if (constraint < STRIDE_FACTOR_MORE_THAN_RES*hp.nside2resol(nside)):
                      print("Warning: pixel resolution constraint not met for nside: ", nside,
                            ", patch_width: ", patch_width,
                            ", max_period_deg: ", max_period_deg,
                            ", num_periods: ", num_periods,
                            ", constraint: ", constraint,
                            )
                      # skip loop iteration
                      continue
                  inner_list8.append([])
                  inner_list9 = inner_list8[-1]
                  for norm_not_product in [True, False]:
                   inner_list9.append([])
                   inner_list10 = inner_list9[-1]
                   for alm in [False]:
                    inner_list10.append([])
                    inner_list11 = inner_list10[-1]
                    for alm_do_filter in [True]:
                     inner_list11.append([])
                     inner_list12 = inner_list11[-1]
                     for alm_fold in [True]:
                      inner_list12.append([])
                      inner_list13 = inner_list12[-1]
                      for std_factor in [1,2,4,8,16]:
                       inner_list13.append([])
                       inner_list14 = inner_list13[-1]
                       for rescale_range in [False]:
                        inner_list14.append([])
                        inner_list = inner_list14[-1]
                        inner_list.append(fnl_1)
                        # Measure execution
                        start_time = datetime.now()

                        sample_fnls = []

                        collect_options = {'num_samples':num_samples,
                               'sample_multiplier':sample_multiplier,
                               'patches_to_average':patches_to_average,
                               'nside':nside,
                               'patch_width':patch_width,
                               'max_period_deg':max_period_deg,
                               'num_periods':num_periods,
                               'norm_not_product':norm_not_product,
                               'alm':alm,
                               'alm_do_filter':alm_do_filter,
                               'alm_fold':alm_fold,
                               'std_factor':std_factor,
                                'rescale_range':rescale_range,
                                'rpts':num_repeats,
                                'rpts_sky':num_repeats_sky} #leaving out only fnl_0 and 1 on purpose

                        write_folder = "../output/" + make_string_from_options(collect_options, fnl_0=fnl_0,
                                                                fnl_1=fnl_1)
                        # register all parameters
                        print_params = make_text_from_options(**{"fnl_0":fnl_0, "fnl_1":fnl_1},
                                                              **collect_options)

                        total_num_homology_points_0 = 0
                        total_skipped_points_0 = 0

                        for repeat in tqdm(range(num_repeats),"Generating Sample Set"):

                          # homology_points_0, found_pixel_ids, skipped_0 = \ #took out found_pixel_ids return and in
                          homology_points_0, skipped_0 = \
                             generate_sample_set(fnl=fnl_0, sky_map=scaled_sky_map_fnl_0,
                                                 **collect_options)

                          total_num_homology_points_0 += len(homology_points_0)
                          total_skipped_points_0 += skipped_0

                          homology_points_1, skipped_1 = \
                             generate_sample_set(fnl=fnl_1, sky_map=scaled_sky_map_fnl_1,
                                                 **collect_options)

                          sample_fnls.append([homology_points_0, homology_points_1])

                        print("fnl=0: Skipped:", total_skipped_points_0, "to produce",
                        total_num_homology_points_0, "points. Rejection rate:",
                        total_skipped_points_0/(total_skipped_points_0 + total_num_homology_points_0))

                        inner_list.append(total_num_homology_points_0//num_repeats)
                        sample_fnls = np.array(sample_fnls)

                        # MIXING!!!! take fnl0 points and mix into fnl1
# sample_fnls[:,1,num_samples//2:,:] = \
                        #     sample_fnls[:,0,num_samples//2:,:]
                        # visualise_sample_fnls(sample_fnls)

                        # create folder
                        os.makedirs(write_folder, exist_ok=True)

                        # save Homology results
                        # rips_snapshot_file = write_folder + \
                        #                     '/configcmb.' + \
                        #                     'save_CMB_Homology.' + \
                        #                     str(num_repeats) + '_repeats_' + \
                        #                     str(num_fnls) + '_fnls' + \
                        #                     '.pickle'

                        # rips_snapshot_file = write_folder + "/data_pis_and_labels.pickle"
                        rips_snapshot_file = None
                        data_PIs, data_labels = \
                                  calc_rips_complex_for_multiple_fnl(
                                      sample_fnls,
                                      fnls=[0, 1], # labels, not values
                                      rips_dim=rips_dim_max,
                                      from_snapshot=None,
                                      to_snapshot=rips_snapshot_file)

                        if do_sim:
                            # save Permutation Test results
                            # perm_snapshot_file = write_folder + \
                            #                      '/configcmb.save_CMB_Perm_Test.pickle'

                            for betti in range(rips_dim_post_process_start, rips_dim_max):

                              # added p_val
                              p_val = permutation_test(data_PIs,
                                                       data_labels,
                                                       num_repeats=num_repeats,
                                                       test_betti=betti,
                                                       visualise=None)
                                                       # to_snapshot=perm_snapshot_file)        

                              inner_list.append((betti, p_val))
                              # save results to parameter file for each betti
                              print_params = print_params + \
                                             "Betti " + \
                                             str(betti) + \
                                             " p_val: " + \
                                             str(p_val) + "\n"

                        tilt_PIs(data_PIs) #tilt all bettis
                        write_PI_diagrams_for_multiple_fnl(data_PIs, data_labels,
                                                           write_folder=write_folder + "/PI_diags_Sim/")

                        if do_sim:
                            # classify and plot
                            for betti in range(rips_dim_post_process_start, rips_dim_max):

                               # save Bayes Test results
                               # bayes_snapshot_file = write_folder + \
                               #                '/configcmb.' + \
                               #                '.save_CMB_Bayes_Test.betti.' + \
                               #                str(betti) + \
                               #                '.pickle'

                               # area under curve
                               auc_value = bayesian_classification(data_PIs,
                                        data_labels,
                                        test_betti=betti,
                                        string_filename_suffix =
                                        "_num_samples_" + str(num_samples) +
                                        "_num_repeats_" + str(num_repeats) +
                                        "_fnls_" + str(fnl_0) + "_and_" + str(fnl_1),
                                        visualise=None,
                                        write_folder=write_folder)
                                        # to_snapshot=bayes_snapshot_file)

                               inner_list.append((betti, auc_value))
                               # save results to parameter file for each betti
                               print_params = print_params + \
                                               "Betti " + \
                                               str(betti) + \
                                               " AUC: " + \
                                               str(auc_value) + "\n"

                            # ML_train_test(data_PIs, data_labels, rips_dim=2)

                        if do_real_sky:
                            mask = hp.ma(hp.read_map(configcmb.full_path_sky_mask)) # 0 is mask, 1 is clear
                            # mask = np.logical_not(mask)
                            scaled_sky_map_real = read_CMB_map(configcmb.full_path_sky_map_2)
                            total_num_homology_points_sky = 0
                            total_skipped_points_sky = 0
                            sample_sky = []
                            collect_options['sample_multiplier']  = 2
                            for repeat in tqdm(range(num_repeats_sky),"Generating Real Sky Sample Set"):
                                homology_points_sky, skipped_sky = generate_sample_set(
                                    **collect_options,
                                    sky_map=scaled_sky_map_real, mask=mask)
                                total_num_homology_points_sky += len(homology_points_sky)
                                total_skipped_points_sky += skipped_sky
                                sample_sky.append([homology_points_sky])

                            data_PIs_sky, _ = calc_rips_complex_for_multiple_fnl(
                                sample_sky,
                                fnls=[-1], # labels
                                rips_dim=rips_dim_max,
                                from_snapshot=None,
                                to_snapshot=None)

                            tilt_PIs(data_PIs_sky) #tilt all bettis

                            write_PI_diagrams_for_multiple_fnl(data_PIs_sky,
                                    -1*np.ones_like(data_PIs_sky),
                                    write_folder=write_folder + "/PI_diags_sky/")

                            # classify_betti = 1
                            for betti in range(rips_dim_post_process_start, rips_dim_max):
                                label, vote, bayes_factors = bayesian_classification(
                                    [data_PIs[betti] + data_PIs_sky[betti]],
                                    [data_labels[betti] + [-1]*num_repeats_sky],
                                    test_betti=0,
                                    number_to_classify=num_repeats_sky,
                                    # string_filename_suffix =
                                    # "_num_samples_sky_" + str(num_samples) +
                                    # "_num_repeats_sky_" + str(repeat_real_sky),
                                    visualise=None)#,
                                    # write_folder=write_folder)
                                    # to_snapshot=None)

                                print("Label: ", label)
                                print("Vote: ", vote)
                                print("Bayes Factor: ", bayes_factors)
                                inner_list.append((betti, label, vote, bayes_factors))
                                print_params = print_params + "\nBetti " + str(betti) + ": " \
                                    + "\n Label: " + str(label) \
                                    + "\n Vote: "  + str(vote) \
                                    + "\n Bayes Factors: " + str(bayes_factors) + "\n"

                            time_elapsed = datetime.now() - start_time 

                            print('Elapsed (hh:mm:ss:ms) {}'.format(time_elapsed))

                            # save timing results to parameter file
                            print_params = print_params + \
                                'Total Runtime (hh:mm:ss:ms) {}'.format(time_elapsed) + "\n"

                        # finally write out all save options and results
                        print(print_params,
                              file=open(write_folder + '/run_parameters_results.txt', 'w'))

    import secrets
    with open("../output/final_output_" + secrets.token_urlsafe(8) + ".pickle", 'wb') as f:
        pickle.dump(output_list, f)

    # with open("final_output_", 'rb') as f:
    #     output_array = pickle.load("final_output_.pickle")
        
    # output_array = np.array(output_list, dtype=object).reshape(-1, 4)
    # num_sample_list = output_array[:,1]
    # auc_list = [auc_tuple[1] for auc_tuple in output_array[:,3]]
    # # plot.plot(num_sample_list, auc_list)
    # # plot.show()
####################################

                    # sky_map=read_CMB_map(configcmb.full_path_sky_map_2)
                    # # sky_map = hp.read_map(configcmb.full_path_sky_map_2)
                    # hp.mollview(sky_map)
                    # plot.show()

                    # mask = hp.ma(hp.read_map(configcmb.full_path_sky_mask))
                    # mask = np.logical_not(mask)
                    # hp.mollview(mask)
                    # plot.show()
