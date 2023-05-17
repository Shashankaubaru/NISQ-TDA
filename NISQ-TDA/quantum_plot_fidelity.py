#from quantum_homology import compare_counts_hellinger
#from qiskit.quantum_info import hellinger_fidelity as compare_counts_hellinger
from qiskit.quantum_info import hellinger_distance as compare_counts_hellinger
import glob

from quantum_homology import edge_2_qasm, square_4_qasm, cube_8_qasm
from quantum_homology import edge_2_honeywell, square_4_honeywell, cube_8_honeywell
from quantum_homology import merge_counts, calc_circuit_depths
from quantum_homology import help_setting_laplacian_arguments

import matplotlib.pyplot as plot
import numpy as np

import pdb

# edge_2_noise_sim = \
#     "../output/qasm_simulator_edge_2_True_True_r_None_edge_only_vec_rccx_RwDimoUXmko.json"
# two_fully_connected_4_noise_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_4_True_True_r_None_edge_only_vec_rccx_1_uInq_0W8Y.json"
# two_fully_connected_8_noise_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_8_True_True_r_None_edge_only_vec_rccx_Elpo1sNBrR4.json"
# two_fully_connected_16_noise_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_16_True_True_r_None_edge_only_vec_rccx_t_QXpOV_EAI.json"

# edge_2_noise_0p1_sim = \
#     "../output/qasm_simulator_edge_2_True_True_r_None_edge_only_vec_rccx0.1_0.1_gpu_*.json"
# two_fully_connected_4_noise_0p1_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_4_True_True_r_None_edge_only_vec_rccx0.1_0.1_gpu_*.json"
# two_fully_connected_8_noise_0p1_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_8_True_True_r_None_edge_only_vec_rccx0.1_0.1_gpu_*.json"
# # two_fully_connected_16_noise_0.1_sim = \
# #     "../output/qasm_simulator_two_fully_connected_clusters_16_True_True_r_None_edge_only_vec_rccx_t_QXpOV_EAI.json"

# edge_2_noise_free_sim = \
#     "../output/qasm_simulator_edge_2_True_True_r_None_edge_only_vec_rccx_vsg7aZN043U.json"
# two_fully_connected_4_noise_free_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_4_True_True_r_None_edge_only_vec_rccx_hz3lnasun1o.json"
# two_fully_connected_8_noise_free_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_8_True_True_r_None_edge_only_vec_rccx_I6pYlKul80M.json"
# two_fully_connected_16_noise_free_sim = \
#     "../output/qasm_simulator_two_fully_connected_clusters_16_True_True_r_None_edge_only_vec_rccx_xkoECQXNHEE.json"



    #####################################
    # KL vals
    #####################################

# raw_data_globs_array = [[edge_2_false_true_edge_only, edge_2_false_true_edge_only_lagos],
#                         [noedge_2_false_true_rccx, noedge_2_false_true_jakarta_rccx],
#                         [tetrahedron_4_false_true, tetrahedron_4_false_true_toronto],
#                         [square_diag_4_false_true_rccx, square_diag_4_false_true_rccx_toronto]]
# # , [square_4_false_true, square_4_false_true_rccx]]

# kl_vals = [0, 0]
# for raw_data_globs in raw_data_globs_array:
#     kl_vals.append(load_merge_compare(*(map(glob.glob, raw_data_globs))))

# kl_vals = np.array(kl_vals).reshape(3, 2)    
# plot.plot(kl_vals)
# plot.show()

# raw_data_globs = [edge_2_qasm, edge_2_honeywell]
# edge_dist = \
#     compare_counts_hellinger(merge_counts(glob.glob(raw_data_globs[0])),
#                              merge_counts(glob.glob(raw_data_globs[1]),
#                                           template=[1,1,1,2]))

# raw_data_globs = [square_4_qasm, square_4_honeywell]
# square_dist = compare_counts_hellinger(merge_counts(glob.glob(raw_data_globs[0])),
#                                       merge_counts(glob.glob(raw_data_globs[1]),
#                                                    template=[2, 6, 6, 4]))

# raw_data_globs = [cube_8_qasm, cube_8_honeywell]
# cube_dist = compare_counts_hellinger(merge_counts(glob.glob(raw_data_globs[0])),
#                                      merge_counts(glob.glob(raw_data_globs[1]),
#                                                   template=[3, 28, 28, 8]))

noise_free_counts_full = []
noise_free_counts_top_10_keys = []
noise_free_counts = []
top_sizes = []
fraction_to_keep = 0.1

#top_sizes = [2**5, 2**18, 2**67, 2**260]
#top_sizes = [2**2, 2**4, 2**8, 2**16]
#top_sizes = [10, 10, 10, 10]
#clip_size = [0, 20, 10, 1]


vertices = [4, 8, 16]
# vertices = [16]
noise_lvls = range(1,4)

for num_vertices in vertices:
    _,_,_,_,_,_,job_name = help_setting_laplacian_arguments(num_vertices=num_vertices,
                                                            backend_name = "aer_simulator_statevector_gpu",
                                                            noise_lvl=0, shots=None)
    noise_free_glob = "../output/" + job_name + "_*.json"
    noise_free_counts_full.append(merge_counts(glob.glob(noise_free_glob), clip_val=1))

    tot_counts = len(noise_free_counts_full[-1])
    top_sizes.append(max(4,int(tot_counts*fraction_to_keep)))
    noise_free_counts_top_10_keys.append([label for label,_ in sorted(noise_free_counts_full[-1].items(),
                                                                 key=lambda item: item[1])[-top_sizes[-1]:]])
    noise_free_counts.append(merge_counts(glob.glob(noise_free_glob),
                                               keep_keys=noise_free_counts_top_10_keys[-1]))

# noise_free_counts =  noise_free_counts_full
# noise_free_counts_top_10_keys = [None for _ in vertices]

noise_vals = []
dists = []
noisy_counts = []

for noise_lvl in noise_lvls:
    noisy_counts_pernoise = []
    dists_pernoise = []
    for i, num_vertices in enumerate(vertices):
        _,_,_,_,_, noise_val, job_name = help_setting_laplacian_arguments(num_vertices=num_vertices,
                                                                backend_name = "aer_simulator_statevector_gpu",
                                                                noise_lvl=noise_lvl, shots=None)
        noise_glob = "../output/" + job_name + "_*.json"
        print(noise_glob)
        noisy_counts_pernoise.append(merge_counts(glob.glob(noise_glob),
                                                  keep_keys=noise_free_counts_top_10_keys[i]))
        dists_pernoise.append(
            compare_counts_hellinger(noise_free_counts[i], noisy_counts_pernoise[-1]))
        # clip_val=clip_size[i])))
    noise_vals.append(noise_val)
    dists.append(dists_pernoise)
    noisy_counts.append(noisy_counts_pernoise)

plot.clf()
# depths = [0, 30, 100, 300]
# depths_True_True_Honeywell1 = np.array(calc_circuit_depths(log_num_vertices_end=4, mid_circuit=True,
#                                                     sqrt=True, do_rccx=True,
#                                                     backend_name_or_Fake='H1-1E'))
#depths = np.concatenate(([0], depths_True_True_Honeywell1[:, 2]))
depths_True_True = np.array(calc_circuit_depths(log_num_vertices_start=2,
                                                log_num_vertices_end=5, mid_circuit=True,
                                                sqrt=True, do_rccx=True,
                                                backend_name_or_Fake='qasm_simulator'))

depths = np.concatenate(([0], depths_True_True[:len(vertices), 2])) # only correct depth selection if vertices in order

print(depths)
fidelities = np.hstack(([[0] for _ in noise_lvls], np.array(dists))) # [0, edge_dist, fully_4_dist, fully_8_dist, fully_16_dist]
print(fidelities)

fig, ax = plot.subplots()
ax.plot(depths, fidelities.T)
ax.legend(list(map(lambda x: "Noise_" + str(x), noise_vals)))
plot.tight_layout()
plot.savefig("../output/hellinger_distance_vs_depth_top_10_clip_noise_free_1_4-16_vertices.png")


# for i in list(noisy_counts[-1][-1].items())[:100]:
#     print(i)

print("Noise free counts full: ", len(noise_free_counts_full[-1]))
print("Noise free counts top: ", len(noise_free_counts[-1]))

noisy_counts_16_3 = list(noisy_counts[-1][-1].items())
noisy_counts_16_2 = list(noisy_counts[-2][-1].items())
noise_free_counts_list = list(noise_free_counts[-1].items())

print(len(noisy_counts_16_3))

min_counts = min(len(noisy_counts_16_2), len(noisy_counts_16_3))
for i in range(min_counts):
    print(noisy_counts_16_2[i])
    print(noisy_counts_16_2[i] == noisy_counts_16_3[i])

allsame = True
for i in range(min_counts):
    # print(noisy_counts_16_2[i])
    ithtruth = (noisy_counts_16_2[i] == noisy_counts_16_3[i])
    if not ithtruth:
        print(noisy_counts_16_2[i], "\n\n", noisy_counts_16_3[i])
        allsame = False

print(allsame)
