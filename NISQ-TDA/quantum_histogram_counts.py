import matplotlib.pyplot as plot
from qiskit.visualization import plot_histogram
import glob

from quantum_homology import edge_2_qasm, square_4_qasm, cube_8_qasm
from quantum_homology import edge_2_honeywell, square_4_honeywell, cube_8_honeywell

from quantum_homology import merge_counts

square_4_honeywell_deg_1_sim = \
        "./Count_square_sim_500_shots_deg1_vec1.json"
square_4_honeywell_deg_2_sim = \
        "./Count_square_sim_500_shots_deg2_vec1.json"


#####################################
# Plot Histogram of Counts
#####################################

# raw_data_globs = [edge_2_qasm, edge_2_honeywell]
# all_counts = [merge_counts(glob.glob(raw_data_globs[0])),
#               merge_counts(glob.glob(raw_data_globs[1]),
#                            template=[1,1,1,2])]

# raw_data_globs = [square_4_qasm, square_4_honeywell]
# all_counts = [merge_counts(glob.glob(raw_data_globs[0])),
#               merge_counts(glob.glob(raw_data_globs[1]),
#                            template=[2, 6, 6, 4])]

# print(expectation_of_laplacian_json(square_4_honeywell_deg_1_sim, 4, mid_circuit=True))
# print(expectation_of_laplacian_json(square_4_honeywell_deg_2_sim, 4, power=2, mid_circuit=True))

raw_data_globs = [cube_8_qasm, cube_8_honeywell]
all_counts = [merge_counts(glob.glob(raw_data_globs[0]), clip_val=1000),
              merge_counts(glob.glob(raw_data_globs[1]),
                           template=[3, 28, 28, 8], clip_val=2)]

# print(sum(all_counts[0].values()), sum(all_counts[1].values()))
# print(all_counts)
# # raw_data_globs = [cube_8_qasm, cube_8_honeywell]
labels = ["QASM sim", "Honeywell"]


# #####################################
# # Anysize raw_data_globs
# #####################################

# # all_counts = list(map(lambda x: merge_counts(glob.glob(x), clip_val=0), raw_data_globs))

# # Make sure, hardware has all keys present in sim:
# # for i in all_counts[0].keys():
# #     all_counts[1][i] = \
# #         all_counts[1].get(i, 0)
# # all_counts = all_counts + [honeywell_formatted_counts]

# #####################################
# # Size 2 raw_data_globs: First simulation, second hardware (allows "pinning" to sim)
# #####################################
# # sim_counts = merge_counts(glob.glob(raw_data_globs[0]))
# # hardware_counts = merge_counts(glob.glob(raw_data_globs[1]))#, clip_val=20)#keep_keys=sim_counts)
# # all_counts = [sim_counts, hardware_counts]

print(sorted(all_counts[0].items(), key=lambda item: item[1])[-20:])
print(sorted(all_counts[1].items(), key=lambda item: item[1])[-20:])
fig, ax = plot.subplots(figsize=(16, 16))
plot_histogram(all_counts,
               #                   number_to_keep=20,
               sort='value',
               title="Sim vs Hardware Runs",
               legend=labels,
               bar_labels=False,
               ax=ax)

xlabels = ax.get_xticklabels()
ax.set_xticklabels(xlabels, rotation=90, fontdict={'fontsize':10})
# ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
# ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
# ax.autoscale(tight=True)
# ax.autoscale_view()
# matplotlib.pyplot.tight_layout.auto_adjust_subplotpars(fig, ax)
# Cube:
# ax.set_ylim(0, 0.05)
ax.set_xticklabels(list(range(len(xlabels))), fontdict={'fontsize':10})
plot.tight_layout()

# plot.savefig("../output/histogram_compare" + raw_data_globs[0][10:-5] + "_" + raw_data_globs[1][10:-5] +  "_new_runs_no_long_ticks.png")

plot.show()


# # noedge_2_false_true_hanoi_rccx = "../output/ibm_hanoi_noedge_2_False_True__vec_rccx_*.json"
# noedge_2_false_true_casablanca_ccx = \
#     "../output/ibmq_casablanca_noedge_2_False_True_uniform_vec_ccx_*.json"
# noedge_2_false_true_uniform = \
#     "../output/ibmq_qasm_simulator_noedge_2_False_True_uniform_vec_rccx_*.json"

# noedge_2_false_true_edge_only_rccx = \
#     "../output/ibmq_qasm_simulator_noedge_2_False_True_edge_only_vec_rccx_*.json"
# # noedge_2_false_true_ionq_edge_only_rccx = \
# #     "../output/ionq_noedge_2_False_True_edge_only_vec_rccx_*.json"
# noedge_2_false_true_ionq_edge_only_rccx = \
#     "../output/ionq_qpu_noedge_2_False_True_edge_only_vec_rccx*.json"
# noedge_2_false_true_jakarta_edge_only_rccx = \
#     "../output/ibmq_jakarta_noedge_2_False_True_edge_only_vec_rccx_*.json"
# noedge_2_false_true_casablanca_edge_only_vec_rccx = \
#     "../output/ibmq_casablanca_noedge_2_False_True_edge_only_vec_rccx_*.json"

# noedge_2_true_true_edge_only_rccx = \
#     "../output/ibmq_qasm_simulator_noedge_2_True_True_r_None_edge_only_vec_rccx_*.json"
# noedge_2_true_true_bogota_rccx = \
# "../output/ibmq_bogota_noedge_2_True_True_r_None_edge_only_vec_rccx_*.json"


# edge_2_false_true_edge_only = \
#     "../output/ibmq_qasm_simulator_edge_2_False_True_edge_only_vec_rccx_*.json"
# edge_2_false_true_edge_only_lagos = \
#     "../output/ibm_lagos_edge_2_False_True_edge_only_vec_rccx_*.json"
# edge_2_false_true_edge_only_ionq = \
#     "../output/ionq_qpu_edge_2_False_True_edge_only_vec_rccx*.json"

# edge_2_false_true_uniform = "../output/ibmq_qasm_simulator_edge_2_False_True_uniform_vec_*.json"
# edge_4_false_true_uniform = "../output/ibmq_qasm_simulator_edge_4_False_True_uniform_vec_*.json"
# # edge_8_false_true = "../output/ibmq_qasm_simulator_edge_8_False_True_uniform_vec_*.json"

# edge_2_true_true = "../output/ibmq_qasm_simulator_edge_2_True_True_r_None_uniform_vec_*.json"
# edge_4_true_true = "../output/ibmq_qasm_simulator_edge_4_True_True_r_None_uniform_vec_*.json"
# # edge_8_true_true = "../output/ibmq_qasm_simulator_edge_8_True_True_r_None_uniform_vec_*.json"

# edge_2_true_true_santiago_r_2 = \
#     "../output/ibmq_santiago_edge_2_True_True_r_2_uniform_vec_*.json"

# edge_2_true_true_santiago_r_2 = \
#     "../output/ibmq_santiago_edge_2_True_True_r_2_uniform_vec_*.json"

# edge_2_false_true_sydney = \
#     "../output/ibmq_sydney_edge_2_False_True_uniform_vec_*.json"

# edge_4_false_true_sydney = \
#     "../output/ibmq_sydney_edge_4_False_True_uniform_vec_*.json"

# edge_4_true_true_lagos = \
#     "../output/ibm_lagos_edge_4_True_True_r_None_uniform_vec_*.json"

# edge_2_true_true_casablanca = \
#     "../output/ibmq_casablanca_edge_2_True_True_r_None_uniform_vec_*.json"

# edge_2_false_true_casablanca_edge_only_rccx = \
# "../output/ibmq_casablanca_edge_2_False_True_edge_only_vec_rccx_*.json"

# square_4_false_true_uniform_vec_rccx = \
#     "../output/ibmq_qasm_simulator_square_4_False_True_uniform_vec_*.json"

# square_4_false_true_casablanca_edge_only_vec_rccx = \
#     "../output/ibmq_casablanca_tetrahedron_4_True_True_r_None_edge_only_vec_rccx_*.json"

# pyramid_8_false_true = "../output/ibmq_qasm_simulator_pyramid_8_False_True_uniform_vec_*.json"
# pyramid_8_false_true_sydney = "../output/ibmq_sydney_pyramid_8_False_True_uniform_vec_*.json"

# square_4_false_true = "../output/ibmq_qasm_simulator_square_4_False_True_uniform_vec_*.json"
# square_4_false_true_sydney = "../output/ibmq_sydney_square_4_False_True_uniform_vec_*.json"

# tetrahedron_4_false_true = \
#     "../output/ibmq_qasm_simulator_tetrahedron_4_False_True_uniform_vec_*.json"
# tetrahedron_4_false_true_toronto = \
#     "../output/ibmq_toronto_tetrahedron_4_False_True_edge_only_vec_rccx_*.json"

# square_4_false_true_rccx = "../output/ibm_cairo_square_4_False_True_uniform_vec_rccx_*.json"

# square_diag_4_false_true_rccx_hanoi = \
#     "../output/ibm_hanoi_square_diag_4_False_True_uniform_vec_rccx_*.json"
# square_diag_4_false_true_rccx = \
#     "../output/ibm_hanoi_square_diag_4_False_True_uniform_vec_rccx_*.json"

# square_diag_4_false_true_rccx_toronto = \
#     "../output/ibmq_toronto_square_diag_4_False_True_edge_only_vec_rccx_*.json"

# square_diag_4_false_true_rccx = \
#     "../output/ibmq_qasm_simulator_square_diag_4_False_True_edge_only_vec_rccx_*.json"

# noedge_2_false_true_honeywell_edge_only_rccx = \
#     "../output/honeywell_033c2e6c1dc5426c9f42475561323b5f.json"

# noedge_4_true_true_rccx = \
#     "../output/ibmq_qasm_simulator_noedge_4_True_True_r_None_edge_only_vec_rccx_B_cSZROgtQQ.json"

# edge_4_true_true_rccx_edge_only_vec = \
#     "../output/ibmq_qasm_simulator_edge_4_True_True_r_None_edge_only_vec_rccx_CnsHh_03Icw.json"

# edge_4_true_true_montreal = \
#     "../output/ibmq_montreal_edge_4_True_True_r_None_edge_only_vec_rccx_43OuFE_B6lc.json"

# edge_4_false_true_montreal = \
#     "../output/ibmq_montreal_edge_4_False_True_edge_only_vec_rccx_e4C-ePg-UjU.json"


# honeywell_counts = json.load(open(noedge_2_false_true_honeywell_edge_only_rccx, "r"))
# print(honeywell_counts.keys())
# honeywell_formatted_counts = {}
# for i in range(5000):
#     measure_string = honeywell_counts['power1splitcproj0'][i] + ' ' + \
#         honeywell_counts['power1complexcproj1'][i] + ' ' + \
#         honeywell_counts['power1complexcproj0'][i] + ' ' + \
#         honeywell_counts['c3'][i]
#     # measure_string = honeywell_counts['power1splitcproj0'][i][-1::-1] + ' ' + \
#     #     honeywell_counts['power1complexcproj1'][i][-1::-1] + ' ' + \
#     #     honeywell_counts['power1complexcproj0'][i][-1::-1] + ' ' + \
#     #     honeywell_counts['c3'][i][-1::-1]

#     honeywell_formatted_counts[measure_string] = \
#         honeywell_formatted_counts.get(measure_string, 0) + 1

# clipped_honeywell_counts ={}
# for i in honeywell_formatted_counts.keys():
#     val = honeywell_formatted_counts[i]
#     if val > 10:
#         clipped_honeywell_counts[i] = val
# honeywell_formatted_counts = clipped_honeywell_counts

# raise(ValueError("Stop, but keep interpreter."))

# raw_data_globs = [edge_2_false_true, edge_4_false_true, edge_2_true_true, edge_4_true_true]
# raw_data_globs = [edge_2_false_true, edge_2_false_true_sydney]
# raw_data_globs = [edge_2_false_true, edge_2_true_true_santiago_r_2]
# raw_data_globs = [edge_4_false_true, edge_4_false_true_sydney]
# raw_data_globs = [edge_4_true_true, edge_4_true_true_lagos]
# raw_data_globs = [edge_2_true_true, edge_2_true_true_casablanca]
# raw_data_globs = [noedge_2_false_true, noedge_2_false_true_casablanca_ccx]
# raw_data_globs = [noedge_2_false_true_rccx, noedge_2_false_true_jakarta_rccx]
# raw_data_globs = [noedge_2_false_true_edge_only_rccx,
#                   noedge_2_false_true_casablanca_edge_only_vec_rccx]
# raw_data_globs = [noedge_2_true_true_rccx, noedge_2_true_true_bogota_rccx]
# raw_data_globs = [noedge_2_false_true_rccx, noedge_2_false_true_ionq]

#     raw_data_globs = [noedge_2_false_true_edge_only_rccx,
#                       noedge_2_false_true_casablanca_edge_only_vec_rccx,
# #                      noedge_2_false_true_jakarta_edge_only_rccx,
# #                      noedge_2_false_true_honeywell_edge_only_rccx,
#                       noedge_2_false_true_ionq_edge_only_rccx]
# labels = ["Sim", "Casablanca", "Jakarta", "Ahem"]

# raw_data_globs = [noedge_4_true_true_rccx]
# raw_data_globs = [edge_4_true_true_rccx_edge_only_vec]
# labels = ["Sim", "Honeywell"]
# raw_data_globs = [edge_4_true_true_rccx_edge_only_vec, edge_4_true_true_montreal]
# labels = ["Sim", "IBM", "Honeywell"]

# raw_data_globs = [edge_4_true_true_rccx_edge_only_vec, edge_4_false_true_montreal]
# labels = ["Sim", "IBM mid_circuit False", "Honeywell"]

# raw_data_globs = [edge_2_false_true_edge_only,
#                   edge_2_false_true_edge_only_lagos,
#                   edge_2_false_true_edge_only_ionq]
# labels = ["Sim", "Lagos", "Ahem"]
