import quantum_homology
from quantum_homology import help_setting_laplacian_arguments, expectation_of_laplacian
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--noise", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  default=0,
                    help="Preset noise levels [0-4], default=0 (no noise)")
parser.add_argument("--shots", type=int, default=50000,
                    help="Number of shots, default=50000")
parser.add_argument("--vertices", type=int, default=4,
                    help="Number of vertices, default=4 (vertices)")
parser.add_argument("--complx", type=int, default=10,
                    help="Preset complex types [1-10], default=10 (two_fully_connected_clusters)")
parser.add_argument("--vec_in", type=int, default=5,
                help="Preset vector wrt which the expectation is taken [1-5], default=5 (fixed rademacher)")
parser.add_argument("--power", type=int, default=1,
                help="Integer power of the Laplacian, default=1 (first moment)")
args = parser.parse_args()

mid_circuit = True
backend_name = 'aer_simulator_statevector_gpu'
# backend_name = 'aer_simulator_statevector'
num_resets = None
do_rccx = True
sqrt = True
rename_job_after_done = True

shots = args.shots
num_vertices = args.vertices
power = args.power

edges_in, min_num_vert, edges_in_type, vec_in_circ, vec_in_circ_type, noise, job_name = \
    help_setting_laplacian_arguments(num_vertices=num_vertices,
                                     mid_circuit=mid_circuit,
                                     do_rccx=do_rccx,
                                     sqrt=sqrt,
                                     num_resets=num_resets,
                                     backend_name=backend_name,
                                     make_edges_in=args.complx,
                                     make_vec_in_circ=args.vec_in,
                                     noise_lvl=args.noise,
                                     shots=shots)

print("Job: ", job_name)

expectation = \
    expectation_of_laplacian(num_vertices=num_vertices,
                             edges_in=edges_in,
                             vec_in_circ=vec_in_circ,
                             power=power,
                             mid_circuit=mid_circuit,
                             sqrt=sqrt,
                             num_resets=num_resets,
                             job_name=job_name,
                             backend_name=backend_name,
                             rename_job_after_done=
                             rename_job_after_done,
                             do_rccx=do_rccx,
                             noise=noise,
                             shots=shots)

print("Done: ", expectation)
