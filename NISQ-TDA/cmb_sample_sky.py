from cmb import read_CMB_map, generate_sample_set, calc_rips_complex_for_multiple_fnl, write_PI_diagrams_for_multiple_fnl
import configcmb
import healpy as hp
import json
import numpy as np
import time
#from memory_profiler import profile
import psutil

#@profile
def calc_PI(homology_points_sky, betti):
    data_PIs_sky, _ = calc_rips_complex_for_multiple_fnl(
          [[homology_points_sky]],
          fnls=[-1], # labels
          rips_dim=betti,
          from_snapshot=None,
          to_snapshot=None)
    

mask = hp.ma(hp.read_map(configcmb.full_path_sky_mask)) # 0 is mask, 1 is clear
# mask = np.logical_not(mask)
scaled_sky_map_real = read_CMB_map(configcmb.full_path_sky_map_2)

collect_options = {'num_samples':64,
                   'sample_multiplier':1,
                   'patches_to_average':1,
                   'nside':1024,
                   'patch_width':32,
                   'max_period_deg':2,
                   'num_periods':2,
                   'norm_not_product':True,
                   'alm':False,
                   'alm_do_filter':False,
                   'alm_fold':None,
                   'std_factor':None,
                    'rescale_range':False,
                    'rpts':1,
                    'rpts_sky':1} #leaving out only fnl_0 and 1 on purpose

homology_points_sky, skipped_sky = generate_sample_set(
    **collect_options,
    sky_map=scaled_sky_map_real, mask=mask)

homology_points_sky = np.array(homology_points_sky)

counts_file = open("../output/cmb_sample_points6.json", "w")
json.dump(homology_points_sky.tolist(), counts_file)
counts_file.close()

times = []
mem = []
calc_PI(homology_points_sky, 1)
    
for betti in range(2,7):

    start_time = time.perf_counter()
    start_mem = psutil.virtual_memory().used
    
    calc_PI(homology_points_sky, betti)
    
    time_elapsed = time.perf_counter() - start_time
    mem_used = psutil.virtual_memory().used - start_mem
    times.append(time_elapsed)
    mem.append(mem_used)

print(times)
print(mem)
    
    # tilt_PIs(data_PIs_sky) #tilt all bettis
    # import pdb
    # pdb.set_trace()

    # data_labels_sky = []
    # for list_of_PIs in data_PIs_sky:   
    #     data_labels_sky.append([-1]*len(list_of_PIs))

    # write_PI_diagrams_for_multiple_fnl(data_PIs_sky,
    #             data_labels_sky,
    #             write_folder="../output/PI_diags_sky_6/",
    #             num_fnls=1)

# # ################### MAIN ################### #
# if __name__ == "__main__":
#     # if not ray.is_initialized():
#     #     ray.init(num_cpus=10)
#     pass
