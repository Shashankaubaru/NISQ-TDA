# noqa: E501,E402,E401,E128
# flake8 disable errors
# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# For Homology
from classical_homology import \
    construct_unfilled_triangle, \
    construct_unfilled_square, \
    complete_unsigned_complex, \
    one_skeleton_triangle, \
    one_skeleton_square, \
    one_skeleton_pyramid, \
    one_skeleton_random, \
    one_skeleton_fully_connected, \
    one_skeleton_tetrahedron, \
    one_skeleton_square_with_diagonal, \
    one_skeleton_unfilled_cube, \
    one_skeleton_n_disconnected_squares

from homology_tools import d_dim_simplices_mask, num_underlying_vertices_vector_length, num_underlying_vertices_simplices, print_signs_simplicial_vector_parts, reshape_vector_to_matrix_and_index, simplices_to_count_vector, count_vector_to_simplices

from math_fiddling import binarize, print_matrix, format_basic, format_complex, print_matrix_and_info, format_real, countBits, num_bits, integer_to_binary_as_string, pair_index_to_two_power_2s, two_power_2s_to_pair_index, num_qubits_from_num_pairs, is_power_of_2, first_non_zero_bit, bin_string2int

#For General Computing
import itertools
import numpy as np
from numpy.linalg import norm as norm
import time
import secrets
#from scipy.spatial.KDTree import query_pairs as find_pairs
from scipy.special import rel_entr, entr
import json
import copy

#For Qiskit use
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import PrimitiveOp, MatrixOp

from qiskit import BasicAer, Aer
import qiskit.providers.aer.noise as qnoise
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import fake_pulse_backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.algorithms import QPE
from qiskit.circuit.library import QFT
from qiskit.aqua.components.initial_states import Custom
#from qiskit.aqua.operators.legacy import MatrixOperator, WeightedPauliOperator
#from qiskit.aqua.operators.legacy import op_converter

from qiskit import AncillaRegister, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.aqua.utils import summarize_circuits
from qiskit.circuit.quantumregister import Qubit
# from qiskit.aqua.circuits import FourierTransformCircuits as Fourier_circ
# from qiskit.circuit.library import QFT
# from qiskit.circuit import Parameter
# from qiskit.aqua.operators import PauliTrotterEvolution, Suzuki

from qiskit.extensions.quantum_initializer.initializer import Initialize

from qiskit.providers.basicaer import UnitarySimulatorPy

from qiskit.aqua import AquaError

from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.utils.controlled_circuit import get_controlled_circuit

from qiskit.aqua.circuits import PhaseEstimationCircuit

from qiskit.aqua.utils.subsystem import get_subsystems_counts

from qiskit.circuit.library import TwoLocal

from qiskit.compiler import transpile, assemble
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.ibmq.managed import IBMQJobManager
from datetime import datetime, timedelta
from dateutil import tz

# from qiskit_ionq import IonQProvider
from qiskit.providers.honeywell import Honeywell

# from qiskit.providers.basicaer import BasicAerJob

import matplotlib.pyplot as plot
from qiskit.visualization import plot_histogram
import glob


# import pdb
# from pdb import set_trace as bp

EPS = 1e-16
JOB_RETRIEVE_LIMIT = 300
JOB_PREVIOUS_DAYS = 20

#######################################################################

############################################
### HARDWARE HELPER FUNCTIONS
############################################

def get_post_select_counts(complete_system_counts, post_select_indices_set=None,
                           post_select_flags_set=None):

    if len(post_select_indices_set)!=len(post_select_flags_set):
        raise ValueError("indices list not same length as flags list.")
    
    mixed_measurements = list(complete_system_counts)
    num_registers = len(mixed_measurements[0].split())
    count_dicts = [{} for _ in range(len(post_select_flags_set))]
    post_selections = [(i,j,k) for i, (j,k) in enumerate(zip(post_select_indices_set,
                                                             post_select_flags_set))]
    for mixed_measurement in mixed_measurements:
        split_measurement = np.array(mixed_measurement.split()[::-1], dtype=str) #-1 reverses so registers are correct, strings themselves remain in qiskit ordering
        for i, post_select_indices, post_select_flags in post_selections:
            if np.all(split_measurement[post_select_indices] == post_select_flags):
                other_indices = list(range(num_registers))
                list(map(other_indices.remove, post_select_indices))
                if other_indices:
                    count_dicts[i][" ".join(split_measurement[other_indices])] = \
                    complete_system_counts[mixed_measurement]
                else:
                    count_dicts[i][mixed_measurement] = complete_system_counts[mixed_measurement]
    return count_dicts # remember other_indices registers are in qiskit ordering

# TODO: make unittests
# get_post_select_counts({'0 0': 10, '1 1' : 20}, [[1],[1]], [['0'],['1']])
# get_post_select_counts({'0 0': 10, '1 1' : 20, '1 2': 30}, [[1],[1]], [['0'],['1']])

def get_fraction_count_for_all_k(complete_system_counts, laplacian, sqrt):

    # 0 simplicies are points, remember, may not be able to count _num_main_qubits in count register, so let's return size num_main_qubits - 1

    # Main
    # main_0_indices = [0]
    # main_0_flags = ['0'*laplacian._num_main_qubits]
    # no, rather let vary
    count_start = 0

    # Ancilla: Not measuring for now! TODO implement error mitigation.
    # count_start = int((laplacian._num_clean_ancillae > 0))
    # ancilla_indices = [1]*count_start #count_start is zero or one only
    # ancilla_flags = ['0'*laplacian._num_clean_ancillae]*count_start
    ancilla_indices = []
    ancilla_flags = []

    # Project Complex
    count_start += 1
    count_end = count_start + laplacian._num_complex_projections
    project_complex_indices = list(range(count_start, count_end))
    project_complex_flags = ['0'*laplacian._num_pairs]*laplacian._num_complex_projections

    # Split
    count_start = count_end
    count_end = count_start + laplacian._num_split_projections
    matching_split_indices_set = [list(range(count_start, count_end))]*\
                                 (laplacian._num_main_qubits-1)
    matching_split_flags_set = []
    for k in range(0, laplacian._num_main_qubits-1): # k is simplex dim
        # k+1 is num vertices:
        k_as_bin_str = integer_to_binary_as_string(k+1, width=laplacian._count_bits_per_projection)
        matching_split_flags_set.append([k_as_bin_str]*laplacian._num_split_projections)

    projection_indices = ancilla_indices + project_complex_indices
    projection_flags = ancilla_flags + project_complex_flags

    projection_indices_set = list(map(lambda matching_split_indices: \
                                       projection_indices + matching_split_indices,
                                       matching_split_indices_set))
    projection_flags_set = list(map(lambda matching_split_flags: \
                                       projection_flags + matching_split_flags,
                                       matching_split_flags_set))

    counts_dicts = get_post_select_counts(complete_system_counts,
                                         projection_indices_set,
                                         projection_flags_set)

    #prob0 = [dict_k['00']/sum(dict_k.values()) for dict_k in prob0_dicts], no forgot about footnote 12, go back to over everything else!
    total_counts = sum(complete_system_counts.values())

    # old confused way, when restricted counts to k simplices
    # frac_counts = []
    # for k, dict_k in enumerate(counts_dicts):
    #     valid_keys = [select_key for select_key in dict_k.keys()
    #                   if sum(map(int, list(select_key))) == k+1]
    #     frac_counts.append(sum([*map(dict_k.get, valid_keys)])/total_counts)

    if sqrt:
        frac_counts = [sum(dict_k.values())/total_counts for dict_k in counts_dicts]
    else:
        frac_counts = [dict_k['0'*laplacian._num_main_qubits]/total_counts for dict_k in counts_dicts]

    return frac_counts

############################################
### BOUNDARY OPERATOR HELPER FUNCTIONS
############################################

def operator_from_pauli_labels(coeff, paulis):
    pauli_ops = []
    for w, p in zip(coeff, paulis):
        pauli_ops += [PrimitiveOp(Pauli.from_label(p), coeff=np.real(w) + np.imag(w)*1.j)]
    return sum(pauli_ops)


# We go straight to the Pauli representation of b + b*
def boundary_operator_plus_conjugate_as_pauli_labels(num_vertices):
    boundary_map_pauli_coeff = [1 for _ in range(num_vertices)]

    first_part = "Z"*(num_vertices-1)
    last_part = "I"*(num_vertices-1)

    boundary_map_pauli_strings = map(lambda i:
                                     first_part[:i] + "X" + last_part[i:],
                                     range(num_vertices))

    return boundary_map_pauli_coeff, list(boundary_map_pauli_strings)


def boundary_operator_hermitian(num_vertices):
    qubit_op = operator_from_pauli_labels(
    *boundary_operator_plus_conjugate_as_pauli_labels(
        num_vertices))
    return qubit_op
#    return (1/np.sqrt(num_vertices))*qubit_op #normalize


def Ri_add(to_add_circ, three_qubits, theta):

    qubit0 = three_qubits[1]
    qubit1 = three_qubits[0]
    target_qubit = three_qubits[2]

    # Yi-1
     # D
    to_add_circ.x(qubit0)
    to_add_circ.s(qubit0)
    to_add_circ.h(qubit0)
     # CNOT
    to_add_circ.cx(qubit0, target_qubit)

    # Xi
    to_add_circ.h(qubit1)
    to_add_circ.cx(qubit1, target_qubit)

    to_add_circ.rz(theta, target_qubit)

    # Xi\dagger
    to_add_circ.cx(qubit1, target_qubit)
    to_add_circ.h(qubit1)

    # Yi-1\dagger
    to_add_circ.cx(qubit0, target_qubit)
    # D\dagger
    to_add_circ.h(qubit0)
    to_add_circ.sdg(qubit0)
    to_add_circ.x(qubit0)

def boundary_operator_unitary_circuit(main_qreg=None, target_register=None, control_qubit=None):

    if isinstance(main_qreg, int):
        main_qreg = QuantumRegister(main_qreg)

    num_vertices = len(main_qreg)
    if target_register is None:
        target_register = QuantumRegister(1)

    boundary_circ = QuantumCircuit(main_qreg, target_register)

    R_circ = QuantumCircuit(main_qreg, target_register)

    #if isinstance(target_qubit, QuantumRegister):
    target_qubit = target_register[0]

    for i in range(1, num_vertices):
        Ri_add(R_circ, [main_qreg[i-1], main_qreg[i], target_qubit],
               np.arctan2(np.sqrt(i), 1))

    # R
    boundary_circ += R_circ
    # P_0
    if control_qubit is None:
        boundary_circ.x(main_qreg[-1])
    else:
        boundary_circ.cx(control_qubit, main_qreg[-1])
    # R\dagger
    boundary_circ += R_circ.inverse()

    return boundary_circ


# Just to check, we can define the boundary operator separately:
def boundary_operator_as_pauli_labels(num_vertices):

    boundary_map_pauli_coeff = [[1/2, 1j/2] for _ in range(num_vertices)]

    first_part = "Z"*(num_vertices-1)
    last_part = "I"*(num_vertices-1)

    boundary_map_pauli_strings = map(lambda i: list(map(
        lambda B01_term: first_part[:i] + B01_term + last_part[i:],
        ["X", "Y"])), range(num_vertices))

    return list(itertools.chain.from_iterable(boundary_map_pauli_coeff)),\
        list(itertools.chain.from_iterable(boundary_map_pauli_strings))

def boundary_operator(num_vertices):
    qubit_op = operator_from_pauli_labels(
    *boundary_operator_as_pauli_labels(num_vertices))
    return qubit_op

# def boundary_operator_conjugate_as_pauli_labels(num_vertices):
#     coeffs, paulis = boundary_operator_as_pauli_labels(num_vertices)
#     for i in range(num_vertices):
#         coeffs[2*i+1] = -coeffs[2*i+1]
#     return coeffs, paulis
#
# def boundary_operator_hermitian_derived_separately(num_vertices):
#     qubit_op = operator_from_pauli_labels(
#     *boundary_operator_conjugate_as_pauli_labels(num_vertices)) \
#     + operator_from_pauli_labels(
#         *boundary_operator_as_pauli_labels(num_vertices))
#     return qubit_op

# def boundary_operator_squared_normalized(num_vertices):
#     # this is automatically hermitian
#     qubit_op = boundary_operator_hermitian(num_vertices)
#     qubit_op *= (1/num_vertices)*qubit_op
#     qubit_op.simplify()
#     return qubit_op

# def boundary_reworked_to_kernel(num_vertices):
#     qubit_op = boundary_operator_squared_normalized(num_vertices)
#     return WeightedPauliOperator.from_list([Pauli.from_label("I"*(num_vertices+1))]) + (-1)*qubit_op

#Careful, here we take the real part only because we expect only real entries!
def classical_matrix_from_pauli_operator(qubit_op):
    return np.real(qubit_op.to_matrix())

def rbs(circ, pair_reg, angle):
        circ.h(pair_reg)
        circ.cz(pair_reg[0], pair_reg[1])
        circ.ry(-angle, pair_reg[0])
        circ.ry(angle, pair_reg[1])
        circ.cz(pair_reg[0], pair_reg[1])
        circ.h(pair_reg)
        return circ

def parity(circ, reg):
        num_left = len(reg)
        qubit_indices = list(range(num_left))
        while num_left > 1:
                new_qubit_indices = []
                for start_index in range(0, num_left-1, 2):
                        circ.cx(reg[qubit_indices[start_index+1]], reg[qubit_indices[start_index]])
                        new_qubit_indices.append(qubit_indices[start_index])
                if start_index < num_left - 2:
                        new_qubit_indices.append(qubit_indices[-1])
                qubit_indices = new_qubit_indices[:]
                num_left = len(qubit_indices)                        
        return circ

def fbs(circ, reg, i, j, theta):
        if i+2<j:
                temp_circ = QuantumCircuit(reg)
                parity(temp_circ, reg[i+1:j])
                circ += temp_circ.inverse()
        if i+1<j:
                circ.cz(i+1,i)
        rbs(circ, [reg[i], reg[j]], theta)
        if i+1<j:
                circ.cz(i+1,i)
        if i+2<j:
                circ += temp_circ
        return circ

def clifford_loader(circ, reg, theta):
        temp_circ = QuantumCircuit(reg)
        
        num_left = len(reg)
        qubit_indices = list(range(num_left))
        while num_left > 1:
                new_qubit_indices = []
                for start_index in range(0, num_left-1, 2):
                        fbs(temp_circ, reg, qubit_indices[start_index], qubit_indices[start_index+1], theta)
                        new_qubit_indices.append(qubit_indices[start_index])
                if start_index < num_left - 2:
                        new_qubit_indices.append(qubit_indices[-1])
                qubit_indices = new_qubit_indices[:]
                num_left = len(qubit_indices)

        circ += temp_circ.inverse()
        circ.x(reg[0])
        circ += temp_circ

        return circ

#######################################################################


############################################
### PROJECT ONTO COMPLEX HELPER FUNCTIONS
############################################

############################################
### Controlled Permutation (Add 1 And Cycle)
############################################

def permutation_circuit(qr=None, qancilla=None, circ=None):
    # By permutation we mean "add 1 with wrapping". The key insight was seeing the 4x4 permutation matrix as a swapped flipped CNOT! Then for the 8x8: swap rows 000 and 100 and so on for higher order.
    #NB: assumes Qiskit ordering

    if qr is None:
        raise Exception("Need to operate on a quantum register.")

    if circ is None:
        raise Exception("Need to operate on a circuit.")

    num_qubits = len(qr)

    if num_qubits==1:
        circ.x(qr[0])
        return circ
    else:
        if num_qubits > 3:
            if qancilla is None:
                qancilla = QuantumRegister(num_qubits - 2)
                circ.add_register(qancilla)
                print("Warning: adding ancilla ", qancilla.name)

    for i in range(1, num_qubits):

        if i > 1:
            circ.x(qr[i-1])

        if i == 1:
            circ.cx(qr[0], qr[i])
        else:
            if i == 2:
                circ.ccx(qr[0], qr[1], qr[i])
            else:
                circ.mct(qr[0:i], qr[i], qancilla, mode="v-chain")
        if i == (num_qubits - 1):
            circ.x(qr[0:i])
    return qancilla

def diagonal_permutation_circuit(qr=None, angle=None, circ=None):
    #default angle for permutation dagger (+ angle)

    if qr is None:
        raise Exception("Need to operate on a quantum register.")

    if circ is None:
        raise Exception("Need to operate on a circuit.")

    n = len(qr)

    if angle is None:
        angle = 2*np.pi/(1<<n)

    for i in range(0,n):
        #circ.u1(angle*(1<<i), qr[i])
        circ.p(angle*(1<<i), qr[i])

def multiplex_diagonal_permutation_circuit(qcontrol=None, qr=None, angle=None, circ=None): #create control permutation_dagger_permutation circuit. This circuit comes from Shende and Bullock, a multiplexed diagonal is a multiplexed Rz gate. Since we have very special structure to the diagonal elements, the multiplexed recursion is short-circuited!

    if circ is None:
        raise Exception("Need to operate on a circuit.")

    if (qr is None) or (qcontrol is None):
        raise Exception("Need to operate on a quantum registers")

    if (type(qcontrol) != Qubit):
        raise Exception("Need first argument to be the single control qubit.")

    n = len(qr) + 1

    if angle is None:
        angle = 2*np.pi/(1<<(n-1))

    for i in range(0,n-1):
        circ.cx(qcontrol, qr[i])
        #circ.u1(angle*(1<<i), qr[i])
        circ.p(angle*(1<<i), qr[i])
        circ.cx(qcontrol, qr[i])

    #circ.u1(-angle*((1<<(n-1))-1), qcontrol)
    circ.p(-angle*((1<<(n-1))-1), qcontrol)

#    technically need to inverse(), since the algorithm takes desired state to zero but can use circ.mirror(). With mirror we do not take the minus sign of the desired angle, but with inverse we would have had to. Finally, we do not even need to call mirror because recursion is short-circuited and this particular circuit is invariant to mirroring because all the subparts commute: the target of CNOTs between subparts are different and U1 on source commutes with CNOT.

def control_permutation_circuit(qcontrol=None, count_register=None,
                                angle=None, qancilla=None, circ=None, leave_out_W=False, leave_out_V=False):

    if circ is None:
        raise Exception("Need to operate on a circuit.")

    if (count_register is None) or (qcontrol is None):
        raise Exception("Need to operate on a quantum registers")

    if (type(qcontrol) != Qubit):
        raise Exception("Need first argument to be the single control qubit.")

    num_count_qubits = len(count_register)

    if angle is None:
        angle = 2*np.pi/(1 << (num_count_qubits + 1))
        
    ########## AAAAAAAAAAAAAA DOCUMENT!!! This is taking the square root! D^2 is the diagonal of the permutation matrix, D is the square root of D^2

    count_register_flipped = count_register[::-1]

    # iqft_inst = QFT(num_count_qubits, inverse=True,
    #                do_swaps=False).to_instruction()
    iqft_inst =  QFT(num_count_qubits, do_swaps=False).inverse().to_instruction()
    qft_inst = QFT(num_count_qubits, inverse=False,
                   do_swaps=False).to_instruction()

    # iqft_inst =  QFT(num_count_qubits, do_swaps=False).inverse().to_instruction()
    # qft_inst = QFT(num_count_qubits, inverse=False,
    #                do_swaps=True).to_instruction()

    # W in paper
    if not leave_out_W:
        qancilla = permutation_circuit(count_register, qancilla, circ=circ)

        #    Fourier_circ.construct_circuit(circ, count_register,
        # inverse=True, do_swaps=False)

        circ.append(iqft_inst, count_register_flipped)#count_register)

        #In the forward Fourier circuit, the swaps come at the beginning, for the inverse, the swaps come at the end. Switching off the swaps implies that after this inverse circuit, the ordering significance is now swapped and subsequent placement of gates must be flipped.
    #    diagonal_permutation_circuit(count_register_flipped, angle=angle, circ=circ)

        diagonal_permutation_circuit(count_register_flipped, angle=angle, circ=circ)
        #gates are flipped because one swap is missing (the significance is swapped)

    # (D D^) in paper
    #multiplex_diagonal_permutation_circuit(qcontrol, count_register_flipped, angle=angle, circ=circ)
    multiplex_diagonal_permutation_circuit(qcontrol, count_register_flipped, angle=angle, circ=circ)
    #gates still flipped because one swap is missing

    # V in paper
    # Fourier_circ.construct_circuit(circ, count_register, inverse=False, do_swaps=False)

    if not leave_out_V:
        circ.append(qft_inst, count_register_flipped) # count_register)

    #with swaps off at the beginning, the flipped significance is already in the flipped order required for the subsequent placement of gates, thereafter everything is back to normal

    return qancilla

##################################################

# def rccx(circ, qubit0, qubit1, target_qubit):
#     circ.ry(np.pi/4, target_qubit)
#     circ.cx(qubit1, target_qubit)
#     circ.ry(np.pi/4, target_qubit)
#     circ.cx(qubit0, target_qubit)
#     circ.ry(-np.pi/4, target_qubit)
#     circ.cx(qubit1, target_qubit)
#     circ.ry(-np.pi/4, target_qubit)

def num_complex_projections(powers=[1], sqrt=False):
    sum_projections = 0

    coeff = 1 if sqrt else 2

    for p in powers:
        sum_projections += coeff*p + 1
    return sum_projections

# flag register: n/2 qubits
# Code up n-1 rounds of n/2 CCNOTS. Placing the gate only if the pair under consideration is not (is) present in edges_in.
# mid-circuit measurement on the flag register
# Bail out if flag register is not all ones (zeros)

def project_onto_complex(circ, edges_in, simplex_qreg, simplex_in, measure_reg=None, num_resets=1,
                         do_rccx=True):
    #simplex_in flag register of size n/2
    #measure_reg size n C 2
    #edges_in are pairs of points. points are identified by their point index (starting at 0)
    #edges_in must be list of list

    num_main_qubits = len(simplex_qreg)
    num_pairs = num_main_qubits*(num_main_qubits - 1)//2
    half_num_main_qubits = num_main_qubits//2

    if measure_reg is not None:
        if (half_num_main_qubits != len(simplex_in)):
            raise ValueError("Size of simplex_in register, "
                             "does not equal n/2.")
    else:
        if (num_pairs != len(simplex_in)):
            raise ValueError("Size of simplex_in register, "
                             "does not equal num_pairs.")

    if not isinstance(edges_in, list):
        raise ValueError("edges_in not a list of a list of two ints.")

    for pair in edges_in:
        if not isinstance(pair, list)  or len(pair) != 2:
            raise ValueError("edges_in does not contain lists of two ints.")
        if not isinstance(pair[0], int)  or not isinstance(pair[1], int):
            raise ValueError("edges_in inner lists do not contain ints.")

    top_qubits = np.arange(num_main_qubits)
    bottom_qubits = np.arange(num_main_qubits)
    bottom_qubits = np.roll(bottom_qubits,-1)
    # gates = []

    for count_round in range(1, half_num_main_qubits+1): # R from spreadsheet
        count_round_from_zero = count_round - 1

        pair_qubits = np.array([*zip(top_qubits, bottom_qubits)])

        take = 1 << (first_non_zero_bit(count_round))

        #the last round only has one inner_round
        if count_round < half_num_main_qubits:
            rounds = [0, 1]
        else:
            rounds = [0]

        for inner_round in rounds:
            take_in_a_row = np.array(range(take))
            take_indices = np.zeros((half_num_main_qubits), dtype=int)
            for i in range((half_num_main_qubits)//take):
                take_indices[i*take:(i+1)*take] = \
                    (take_in_a_row + i*(2*take) + inner_round*take) % num_main_qubits

            for pairs_index, pair in enumerate(pair_qubits[take_indices].tolist()):
                if pair not in edges_in:  # consider using bisect search of sorted list
                    pair.reverse()
                    if pair not in edges_in:
                        if measure_reg is not None:
                            qubits = (pair[0], pair[1], simplex_in[pairs_index])
                        else:
                            qubits = (pair[0], pair[1], simplex_in[
                                count_round_from_zero*num_main_qubits +
                                inner_round*half_num_main_qubits + pairs_index])

                        if do_rccx:
                            circ.rccx(*qubits)
                            # rccx(circ, *qubits)
                        else:
                            circ.ccx(*qubits)

            if measure_reg is not None:
                # could use: circ._mid_circuit, but prefer not to delve into internals
                circ.measure(simplex_in, measure_reg[
                    count_round_from_zero*num_main_qubits + inner_round*half_num_main_qubits:
                    count_round_from_zero*num_main_qubits + (inner_round+1)*half_num_main_qubits])
                
                for _ in range(num_resets):
                    circ.reset(simplex_in)

        bottom_qubits = np.roll(bottom_qubits, -1)

    return circ


#######################################################################
### SPLIT BY SIMPLEX DEGREE K (entangle with counter reg, then PROJECT)
#######################################################################


def split_k(circ, from_qreg, counter_qreg, qancilla=None):
    # , reshuffle=False):

    # PROJECTION ONTO K-SIMPLICES

    #warning, code similar to within control_permutation_circuit
    count_register_flipped = counter_qreg[::-1]
    num_count_qubits = len(counter_qreg)
    angle = -2*np.pi/(1 << (num_count_qubits + 1))

    # separate by k for k-simplices
    for i in range(len(from_qreg)):
        
        leave_out_W = True if i > 0 else False
        leave_out_V = True if i < len(from_qreg)-1 else False
        
        qancilla = control_permutation_circuit(from_qreg[i], counter_qreg, qancilla=qancilla,
                                               circ=circ, leave_out_V=leave_out_V,
                                               leave_out_W=leave_out_W)
        if leave_out_V:
            diagonal_permutation_circuit(count_register_flipped, angle=angle, circ=circ)

    return qancilla

    # #make 1 and 2, 1x neighbours, if we measure counter_qreg[1], it collapses the entangled simplices
    # if reshuffle:
    #     circ.cx(counter_qreg[0], counter_qreg[1])

    #interfere neighbours -> not necessary (maybe it could help later?)
    #circ.h(counter_qreg[0])


def split_k_inverse(circ, from_qreg, counter_qreg):
    # TODO: decide what to do about qancilla
    tempCirc = QuantumCircuit(from_qreg, counter_qreg)
    split_k(tempCirc, from_qreg, counter_qreg)

    circ.data += tempCirc.inverse().data

def num_split_projections(powers=[1], sqrt=False):

    sum_projections = 0
    add_proj = 0 if sqrt else 1

    intdiv = 2 if sqrt else 1

    for p in powers:
        sum_projections += p//intdiv + 1
    return sum_projections


#######################################################################


############################################
### OTHER HELPER FUNCTIONS
############################################


def initialize_register(qcirc, complex_vector, qreg, inverse=False):

    circ_complex = QuantumCircuit(qreg)
    myinit = Initialize(complex_vector)
    circ_complex.append(
        myinit.gates_to_uncompute().to_instruction(), qreg)
    if inverse:
        qcirc += circ_complex
    else:
        qcirc += circ_complex.inverse()

def unroll_projections(list_projs):
    unrolled_proj = []
    for projs in list_projs:
            unrolled_proj.extend(projs)
    return unrolled_proj

def advanced_index_selecting_diagonal_of_split_k(laplacian):

    # Old:
    # select_dim = [0, 0]
    # select_dim.extend(list(np.ones(
    #     laplacian._num_complex_projections, dtype=int)
    #                        * ((1 << num_pairs) - 1)))
    # select_dim.extend([np.array(range(1, num_main_qubits + 1), dtype=int)
    #                    for _ in
    #                    range(
    #                     laplacian._num_split_projections)])

    num_main_qubits = laplacian._num_main_qubits
    advanced_index_array_base = np.arange(1, num_main_qubits, dtype=int)
    # This is counting vertices, 1, num_main_qubits+1 is from 1 to num_main_qubits, inclusive. Careful if complex is fully connected, introduce check or wrap % num_main_qubits - 1, or leave out num_main_qubits, yes do this
    advanced_index_array = np.zeros(num_main_qubits - 1, dtype=int)

    for i in range(laplacian._num_split_projections):
        advanced_index_array += \
            advanced_index_array_base*(1 << (i*laplacian._count_bits_per_projection))

    return advanced_index_array

def reshape_into_tensor_of_dim_main_ancilla_proj_split(circ_output, laplacian):

    # Old:
    # reshape_dim = np.zeros(2 + laplacian._num_complex_projections +
    #                        laplacian._num_split_projections, dtype=int)
    # reshape_dim[0] = 1 << num_main_qubits
    # reshape_dim[1] = 1 << laplacian._num_clean_ancillae
    # reshape_dim[2:2+laplacian._num_complex_projections] = \
    #     np.ones(laplacian._num_complex_projections) *\
    #     (1 << num_pairs)
    # reshape_dim[2+laplacian._num_complex_projections:] = \
    #     np.ones(laplacian._num_split_projections) *\
    #     (1 << laplacian._count_bits_per_projection)

    # output_tensor = circ_output.reshape(tuple(reshape_dim),
    #                                      order='F')

    output_tensor = circ_output.reshape(
        1 << laplacian._num_main_qubits,
        1 << laplacian._num_clean_ancillae,
        1 << laplacian._num_complex_projections*laplacian._num_pairs,
        1 << laplacian._num_split_projections* \
        laplacian._count_bits_per_projection,
        order='F')

    return output_tensor

#######################################################################
#######################################################################

############################################################
### MAIN CLASSES TO CREATE CIRCUITS FOR MOMENTS OF LAPLACIAN
############################################################


class Projected_Laplacian(QuantumCircuit):
    # Since Laplacian is implemented using Taylor expansion
    # a lot of post-processing is required
    # leave out first projection because assuming only one simplex already in
    # the complex (Ken) -> yes option open again? -> no, because using
    # Rademacher vectors
    """
     Direct Unitary Method: boundary_operator_unitary_circuit(num_vertices)
    """
    def __init__(self, *, num_vertices=None, edges_in=None, power=1,
                 num_ancillae_eigenvalue_resolution=0,
                 sqrt=False, mid_circuit=False, num_resets=None, do_rccx=True):

        # Registers: _main_qreg, _clean_ancilla, _reusable_qreg OR (_complex_qregs AND _split_qregs)
        # Size: _num_main_qubits, _num_clean_ancillae, ._num_projection_qubits (= _half_num_main_qubits* OR
        # (_num_complex_projections*_num_pairs  +
        # _num_split_projections*_count_bits_per_projection))

        ####################################################
        # Consume num_vertices, edges_in
        ####################################################
        if num_vertices is None:
            raise(ValueError("num_vertices not specified from which to build up complex."))

        if edges_in is None:
            raise(ValueError("edges_in not specified from which to build up complex."))
        
        # TODO: worry about user inputing the wrong size edges_in
        self._edges_in = edges_in
        # self._num_pairs = len(edges_in)
        # self._num_qubits = num_qubits_from_num_pairs(self._num_pairs)
        if not is_power_of_2(num_vertices):
            raise ValueError("For now, num_vertices must be a power of two.")
        self._num_main_qubits = num_vertices # consider getting from largest vertex in edges_in, nah, see remap
        self._half_num_main_qubits = num_vertices // 2
        self._num_pairs = num_vertices*(num_vertices-1)//2
        self._main_qreg = QuantumRegister(self._num_main_qubits, "mainqreg")
        super().__init__(self._main_qreg)

        self._count_bits_per_projection = num_bits(self._num_main_qubits)

        # Should be number + 1, to store the all-vertex simplex. Decided to save that bit (we're in the case of power of 2). We could introduce check for the fully connected one-skeleton and return something meaningful, but rather just say we're not interested in simplices of order num_vertices, which is the case.

        self._num_permutation_ancillae = max(0,
            self._count_bits_per_projection - 2)

        if not mid_circuit:  # and self._num_permutation_ancillae > 0:

            self._num_clean_ancillae = max(self._num_permutation_ancillae, 1)
            # max(# self._num_pairs, # n^2 vs logn, don't need. The 1 is for boundary.

            # in NISQ it may be better to have separate registers to limit
            # noise? or use mid-circuit reset!!
            self._clean_ancilla = AncillaRegister(self._num_clean_ancillae, # could multiply by numsplit and measure all at the end as a noise detection scheme
                                                    name="clean_ancillae")

            self.add_register(self._clean_ancilla)
        else: # for the case mid_circuit=True, we still don't need extra quantum ancillae, even though we could get classical ancillae answers out. TODO: do just that for error mitigation.
            self._num_clean_ancillae = 0
            self._clean_ancilla = None

        ####################################################
        # Consume power, num_ancillae_eigenvalue_resolution
        ####################################################
        self.base_power = power

        self._num_ancillae_eigenvalue_resolution = \
            int(num_ancillae_eigenvalue_resolution)

        if self._num_ancillae_eigenvalue_resolution > 0:
            self.powers = [int(self.base_power*(2**i)) for i in range(
            self._num_ancillae_eigenvalue_resolution)]

            self._control_reg_consolidate = QuantumRegister(1)
            self.add_register(self._control_reg_consolidate)

            if self._num_ancillae_eigenvalue_resolution == 1:
                self._control_reg = None
            else:
                self._control_reg = QuantumRegister(self._num_ancillae_eigenvalue_resolution)
                self.add_register(self._control_reg)
        else:
            self.powers = [int(self.base_power)]
            self._control_reg_consolidate = None

        ####################################################
        # Consume sqrt, mid_circuit
        ####################################################


        self._sqrt = sqrt
        self._mid_circuit = mid_circuit

        self._num_complex_projections = num_complex_projections(self.powers,
                                                                self._sqrt)
        self._num_split_projections = num_split_projections(self.powers,
                                                              self._sqrt)

        if mid_circuit:
            self._reusable_qreg = AncillaRegister(
                max(self._half_num_main_qubits,
                    self._count_bits_per_projection + self._num_permutation_ancillae,
                    1),
                    name="reusableprojqreg")
             # Complex projection (always the max), Order projection (count + mct), boundary circ exponentiate Z on target
            self._num_projection_qubits = len(self._reusable_qreg)

            self.add_register(self._reusable_qreg)

            self._boundary_sum_unitary_circuit = \
                boundary_operator_unitary_circuit(self._main_qreg,
                                                  self._reusable_qreg, # don't worry whole reg not used
                                                  self._control_reg_consolidate)
            # consider reset -> yes see power

        else:
            self._boundary_sum_unitary_circuit = \
                boundary_operator_unitary_circuit(self._main_qreg,
                                                  self._clean_ancilla,
                                                  self._control_reg_consolidate)

            self._complex_qregs = []
            self._split_qregs = []

            for power in self.powers:

                projections_for_power = []
                power_num_complex_projections = num_complex_projections([power], self._sqrt)

                for projection in range(power_num_complex_projections):
                    projections_for_power.append(
                        AncillaRegister(self._num_pairs,
                            name="power" + str(power) +
                                 "complexqproj" + str(projection)))
                self._complex_qregs.append(projections_for_power)

                projections_for_power = []
                power_num_split_projections = num_split_projections([power], self._sqrt)

                for projection in range(power_num_split_projections):
                    projections_for_power.append(
                        AncillaRegister(self._count_bits_per_projection,
                            name="power" + str(power) +
                                 "splitqproj" + str(projection)))
                self._split_qregs.append(projections_for_power)

            self.add_register(*unroll_projections(self._complex_qregs),
                              *unroll_projections(self._split_qregs))

            self._num_projection_qubits = self._num_complex_projections*self._num_pairs + \
                self._num_split_projections*self._count_bits_per_projection

        #technically don't need _main_creg at all, if backend is statevector_simulator, but can't check here, because don't know what backend is, no need to pass because no harm if created.
        if self._sqrt:
            self._main_creg = ClassicalRegister(self._num_main_qubits)
            self.add_register(self._main_creg)
        else:
            self._main_creg = ClassicalRegister(1)
            self.add_register(self._main_creg)

        self._complex_cregs = []
        self._split_cregs = []
        for power in self.powers:
            projections_for_power = []
            for projection in range(num_complex_projections([power],
                                                      self._sqrt)):
                projections_for_power.append(
                    ClassicalRegister(self._num_pairs,
                        name="power" + str(power) +
                             "complexcproj" + str(projection)))
            self._complex_cregs.append(projections_for_power)

            projections_for_power = []
            for projection in range(num_split_projections([power],
                                                      self._sqrt)):
                projections_for_power.append(
                    ClassicalRegister(self._count_bits_per_projection,
                        name="power" + str(power) +
                             "splitcproj" + str(projection)))
            self._split_cregs.append(projections_for_power)

        self.add_register(*unroll_projections(self._complex_cregs),
                          *unroll_projections(self._split_cregs))

        self._num_total_qubits = self._num_main_qubits + self._num_clean_ancillae + \
                                 self._num_projection_qubits

        # Uncomment to use as a form of error detection
        # # put at end (first in count list) to allow for easier removal
        # if (not mid_circuit) and self._num_permutation_ancillae > 0:
        #     self._clean_ancilla_creg = ClassicalRegister(
        #         self._num_permutation_ancillae)
        #     self.add_register(self._clean_ancilla_creg)
        # else:
        #     self._clean_ancilla_creg = None
        #     # should actually peal away case of mid-circuit and possibly measure
        #     # for split_projection as a noise detection mechanism! also reset reusable quantum register

        # Both mid_circuit=True and False need classical registers


        ####################################################
        # Consume num_resets and do_rccx
        ####################################################

        if num_resets is None:
            self._num_resets = int(mid_circuit)
        else:
            self._num_resets = num_resets
            if mid_circuit:
                if num_resets == 0:
                    print("Warning: mid_circuit==True and num_resets==0.")
                elif num_resets > 0:
                    print("Warning: mid_circuit==False but num_resets>0.")

        self._do_rccx = do_rccx

    def measure_all_qregs(self):
        """Only to be called for "count" backends (sim or hardware), not to be called for statevector backends. Place measurements on projection registers if mid_circuit is false. Always place measurements on main register (sqrt uses all counts on main reg, not sqrt uses 0 of main reg)! Technically don't need to place measurements on ancillae, but could be used to check for errors?"""

        if self._sqrt:
            self.measure(self._main_qreg, self._main_creg)
        else:
            self.measure(self._main_qreg[0], self._main_creg[0])

        if not self._mid_circuit:
            for qreg_for_power, creg_for_power in zip(unroll_projections(self._complex_qregs),
                                                        unroll_projections(self._complex_cregs)):
                self.measure(qreg_for_power, creg_for_power)
            for qreg_for_power, creg_for_power in zip(unroll_projections(self._split_qregs),
                                                        unroll_projections(self._split_cregs)):
                self.measure(qreg_for_power, creg_for_power)

            # uncomment to use as a form of error detection also introduce such measurements for
            # mid_circuit=True in the loop
            # if self._num_clean_ancillae > 0:
            #     self.measure(self._clean_ancilla, self._clean_ancilla_creg)

    def split_ancillae(self):
        if self._num_permutation_ancillae > 0:
            return self._clean_ancilla[:self._num_permutation_ancillae]
        else: return None

    def power(self, power, matrix_power=False):
        # Warning returns first entry only, but should only have one and only one
        # Note not storing power, so .power(2).control() won't control the square, rather: .control(power=2) -> reconsider this now that control automatically handled?

        if matrix_power:
            raise Exception("Matrix Power not supported")

        power_index = self.powers.index(power)
        count_complex_projection = 0
        count_split_projection = 0

        def place_complex_projection():
            nonlocal count_complex_projection
            if self._mid_circuit:
                complex_qreg = self._reusable_qreg[:self._half_num_main_qubits] # self._num_pairs]
                complex_creg = self._complex_cregs[power_index][count_complex_projection]
            else:
                complex_qreg = self._complex_qregs[power_index][count_complex_projection]
                complex_creg = None

            project_onto_complex(self, self._edges_in, self._main_qreg,
                                 complex_qreg, complex_creg, self._num_resets, do_rccx=self._do_rccx)

            count_complex_projection += 1

        def place_split_projection():
            nonlocal count_split_projection
            if self._mid_circuit:
                split_qreg = self._reusable_qreg[:self._count_bits_per_projection]
                ancilla_qreg = self._reusable_qreg[self._count_bits_per_projection:]
            else:
                split_qreg = self._split_qregs[power_index][count_split_projection]
                ancilla_qreg = self.split_ancillae()

            split_k(self, self._main_qreg, split_qreg,
                            qancilla=ancilla_qreg) # self.split_ancillae())

            if self._mid_circuit:
                self.measure(split_qreg,
                           self._split_cregs[power_index][count_split_projection])
                for _ in range(self._num_resets):
                    self.reset(split_qreg)
            count_split_projection += 1

        place_complex_projection()
        place_split_projection()

        if self._sqrt:
            number_of_PB_pairs = power
        else:
            number_of_PB_pairs = 2*power

        for i in range(number_of_PB_pairs):
            self += self._boundary_sum_unitary_circuit  # automatically caters for control
            if self._mid_circuit:
                for _ in range(self._num_resets):
                    self.reset(self._reusable_qreg[0])  # boundary circuit uses the first qubit of the reusable register as the ancilla
            place_complex_projection()
            if (i % 2):
                    place_split_projection()

    def control(self, num_ctrl_qubits=None, label=None, ctrl_state=None,
                power=1, use_basis_gates=True):  # TODO: write unit tests and use in QPE
        # overloading, not controlling projection since shouldn't change anything?
        # consider doing the 'wrong' black-box control!

        if num_ctrl_qubits is None:
            num_ctrl_qubits = self._num_ancillae_eigenvalue_resolution

        if label is None:
            label='c_{}'.format(self.name)

        if ctrl_state is not None:
            raise Exception("Not handling ctrl_state=True at the moment.")

        if self._control_reg is not None:
            self.mct(self._control_reg, self._control_reg_consolidate)
        self.power(power)

class Project_and_Split(Projected_Laplacian): # TODO: still to write unit_tests
    def __init__(self, edges_in=None, num_vertices=None, mid_circuit=False):
        super().__init__(edges_in=edges_in,  num_vertices=num_vertices, power=0, mid_circuit=mid_circuit)
        # self.power(0)

class Projected_Boundary(Projected_Laplacian):
    def __init__(self, edges_in=None, num_vertices=None, power=1,
                 num_ancillae_eigenvalue_resolution=0, mid_circuit=False):
        super().__init__(edges_in=edges_in, num_vertices=num_vertices, power=power,
                         num_ancillae_eigenvalue_resolution=num_ancillae_eigenvalue_resolution,
                         sqrt=True, mid_circuit=mid_circuit)

#######################################################################

#######################################################################
# FUNCTIONS TO RUN CIRCUITS and POST-PROCESS
#######################################################################

def print_unitary_and_trace(circ, trace=True):
    usim = BasicAer.get_backend('unitary_simulator')
        # usim = Aer.get_backend('aer_simulator')
        # laplacian.save_unitary()
    temp_circuit = transpile(circ, backend=usim, optimization_level=0)
    qobj = assemble(temp_circuit)
    unitary = usim.run(qobj).result().get_unitary()
    print_matrix(unitary, vecfunc=format_real)
    if trace:
        import pdb
        pdb.set_trace()

def expectation_of_laplacian(*, num_vertices, edges_in, vec_in=None, vec_in_circ=None,
                             power=1, mid_circuit=False, sqrt=True, num_resets=None,
                             job_name=None, backend_name="aer_simulator_statevector",
                             rename_job_after_done=False, do_rccx=True,
                             noise=None, shots=None):

    def rename_job(job):
        nonlocal job_name
        nonlocal backend_local_sim
        nonlocal rename_job_after_done
        job_name = job_name + "_" + secrets.token_urlsafe(8)
        if not backend_local_sim and rename_job_after_done:
            # not isinstance(job, BasicAerJob)
            print("Renaming ibmq job to: ", job_name)
            job.update_name(job_name)

    if vec_in is not None and vec_in_circ is not None:
        raise ValueError("Error: vec_in and vec_in_circ specified. Specify one or neither.")
    # TODO: rather check type of single argument
    else:
        qreg_temp = QuantumRegister(num_vertices)
        vec_circ = QuantumCircuit(qreg_temp)

        if vec_in is not None:
            initialize_register(vec_circ, vec_in, qreg_temp)
        elif vec_in_circ is not None:
            vec_circ = vec_in_circ
        else:
            vec_circ.h(qreg_temp)

        vec_circ_inst = vec_circ.to_instruction()
        vec_circ_inverse_inst = vec_circ.copy().inverse().to_instruction()

    result = None

    if backend_name is None:
        backend_any_asked = True
    else:
        backend_any_asked = False
        backend_local_sim = (backend_name == 'qasm_simulator' or # old python simulator
                             backend_name == 'statevector_simulator' or # old python
                             backend_name == 'aer_simulator_statevector' or # new C++
                             backend_name == 'aer_simulator_statevector_gpu' or # new GPU
                             backend_name == 'aer_simulator_density_matrix' or # new C++
                             backend_name == 'aer_simulator_density_matrix_gpu') # new GPU
        gpu = (backend_name == 'aer_simulator_statevector_gpu' or
               backend_name == 'aer_simulator_density_matrix_gpu')


    if backend_local_sim:
        if mid_circuit or noise is not None:
            if backend_name == 'statevector_simulator':
                raise ValueError('mid_circuit/noise not compatible with old statevector '
                'simulator')    
        backend = Aer.get_backend(backend_name)

        if gpu:
            backend.device="GPU" # should not be needed
            backend.set_options(batched_shots_gpu=True)
            backend.set_options(batched_shots_gpu_max_qubits=32)
            if shots is None:
                shots = 50000
        
        if noise is not None:
           # Error probabilities
            prob_1 = noise[0]  # 1-qubit gate
            prob_2 = noise[1]   # 2-qubit gate
            prob_mat = [[1-prob_2,prob_2],[prob_2,1-prob_2]]

            # Depolarizing quantum errors
            error_1 = qnoise.depolarizing_error(prob_1, 1)
            error_2 = qnoise.depolarizing_error(prob_2, 2)
            read_err = qnoise.ReadoutError(prob_mat)

            # Add errors to noise model
            noise_model = qnoise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            noise_model.add_all_qubit_readout_error(read_err)

            # Get basis gates from noise model
            basis_gates = noise_model.basis_gates
            backend.set_options(noise_model=noise_model, basis_gates=basis_gates)
        else:
            noise_model = None

        # mybackend_shots = 80000
        # backend._configuration.n_qubits = 24
        # if backend_name == 'qasm_simulator':
        #     backend._configuration.max_shots = 2*10**3#10**6 #1000 # 10**7
    else:
        backend = None
        if backend_name == "ionq_qpu":
            provider = IonQProvider("YKI565sxYHK1Pe3w4mRRHOQ6EvhbGXeF")
            # print(provider.backends())
            backend = provider.get_backend("ionq_qpu")
            if shots is None:
                shots = 100 # backend._configuration.max_shots
        elif backend_name == "HQS-LT-S1-SIM" or backend_name == "HQS-LT-S2-SIM" \
             or backend_name == "HQS-LT-S1" or backend_name == "HQS-LT-S2":
            Honeywell.load_account()
            backend = Honeywell.get_backend(backend_name)
            if shots is None:
                shots = 2000
        else:
            if IBMQ.active_account() is None:
                IBMQ.enable_account(\
                                    'ec1952054b99bc0febe3c3d296521da193190fb4f9c3925d7'
                                    'b7a015c443240f07263c4f9b6bdbf1a0c99f6fe7451af029b'
                                    '2c52d06f222e94bc467f14b8e2adf6')
            provider = IBMQ.get_provider(hub='ibm-q-internal') # ('ibm-q-internal')

            if backend_name is not None:
                backend = provider.get_backend(backend_name)

            if job_name is not None:
                past_days = datetime.now() - timedelta(days=JOB_PREVIOUS_DAYS)
                # Add local tz in order to compare to `creation_date` which is tz aware.
                # past_days_tz_aware = past_days.replace(tzinfo=tz.tzlocal())
                if backend is None:
                    myjobs = provider.backends.jobs(limit=JOB_RETRIEVE_LIMIT,
                                                    start_datetime=past_days)
                else:
                    myjobs = backend.jobs(limit=JOB_RETRIEVE_LIMIT,
                                          start_datetime=past_days)

                myjob_dict = {myjob.name():myjob for myjob in myjobs}

                if job_name in list(myjob_dict):
                    myjob_status = myjob_dict[job_name].status()
                    if backend is None:
                        backend = myjob_dict[job_name].backend()
                        backend_name = backend.name()
                    if myjob_status is not JobStatus.DONE:
                        print("Found job: ", job_name, ", Status is: ",
                              myjob_status, ". Rerun later.")
                        return (backend_name, job_name)
                    else:
                        result = myjob_dict[job_name].result()
                        print("Found job: ", job_name, ", Status is: ",
                              myjob_status, ".")
                        rename_job(myjob_dict[job_name])

    # Need the laplacian object to choose hardware backend, run and/or to analyse results

    laplacian = Projected_Laplacian(num_vertices=num_vertices,
                                    edges_in=edges_in, power=power,
                                    mid_circuit=mid_circuit, sqrt=sqrt,
                                    num_resets=num_resets,
                                    do_rccx=do_rccx)
    
    q_main_reg = laplacian._main_qreg
    num_main_qubits = laplacian._num_main_qubits
    num_pairs = laplacian._num_pairs
    num_total_qubits = laplacian._num_total_qubits

    if result is None:
        if backend is None: # when sim is False and no job found
            available_devices = provider.backends(
                filters=lambda x: (x.configuration().n_qubits >= num_total_qubits) and
                (not x.configuration().simulator))  # and x.status().operational==True)
            if not available_devices:
                raise AssertionError("No available devices with at least " +
                                     str(num_total_qubits)
                                     + " qubits.")
            backend = least_busy(available_devices)
            backend_name = backend.configuration().backend_name
            backend_any_asked = True
            print("Least Busy Backend:", backend_name)

        if shots is None:
            shots = backend._configuration.max_shots

        # CIRCUIT CONSTRUCTION
        laplacian.append(vec_circ_inst, qargs=q_main_reg)

        laplacian.power(power) # already taking into account sqrt (during _init_)

        if not sqrt: # here and below could use laplacian._sqrt to emphasise laplacian, chose not to since did assign local variables to other laplcian variables
            laplacian.append(vec_circ_inverse_inst, qargs=q_main_reg)

        # print_unitary_and_trace(laplacian)

        if backend_name != 'statevector_simulator': # may need to check for ibmq statevector_simulator
            laplacian.measure_all_qregs()

            if backend.configuration().n_qubits == 27:
                if num_main_qubits == 2:
                    if mid_circuit:
                        rqreg = laplacian._reusable_qreg
                        initial_layout = {q_main_reg[0]: 11,
                                          q_main_reg[1]: 16,
                                          rqreg[0]: 14}
                    else:
                        cqreg = laplacian._complex_qregs[0]
                        sqreg = laplacian._split_qregs[0]
                        aqreg = laplacian._clean_ancilla
                        initial_layout = {q_main_reg[0]: 11,
                                          q_main_reg[1]: 16,
                                          aqreg[0]: 14,
                                          sqreg[0][0]: 13,
                                          cqreg[0][0]: 8,
                                          cqreg[1][0]: 19}
                elif num_main_qubits >= 4:
                    if mid_circuit:
                        rqreg = laplacian._reusable_qreg
                        initial_layout = {q_main_reg[0]: 11,
                                          q_main_reg[1]: 16,
                                          q_main_reg[2]: 15,
                                          q_main_reg[3]: 10,
                                          rqreg[0]: 14,
                                          rqreg[1]: 12}
                                          #rqreg[2]: 13}
                    else:
                        cqreg = laplacian._complex_qregs[0]
                        sqreg = laplacian._split_qregs[0]
                        aqreg = laplacian._clean_ancilla
                        initial_layout = {q_main_reg[0]: 11,
                                          q_main_reg[1]: 16,
                                          q_main_reg[2]: 15,
                                          q_main_reg[3]: 10,
                                          aqreg[0]: 13,
                                          sqreg[0][0]: 14,
                                          sqreg[0][1]: 12,
                                          cqreg[0][0]: 8,
                                          cqreg[0][1]: 19,
                                          cqreg[0][2]: 18,
                                          cqreg[0][3]: 7,
                                          cqreg[0][4]: 5,
                                          cqreg[0][5]: 22,
                                          cqreg[1][0]: 9,
                                          cqreg[1][1]: 20,
                                          cqreg[1][2]: 17,
                                          cqreg[1][3]: 6,
                                          cqreg[1][4]: 21,
                                          cqreg[1][5]: 4}
            else:
                initial_layout = None
        else:
            initial_layout = None

        # RUN CIRCUIT
        unrolled_laplacian = transpile(laplacian, backend=backend,
                                       seed_transpiler=11, optimization_level=0)
        optimized_laplacian = transpile(laplacian, backend=backend,
                                        seed_transpiler=11, optimization_level=3,
                                        initial_layout=initial_layout)
        print("width: ", optimized_laplacian.width(), "depth: ",
              optimized_laplacian.depth())
 
        if backend_name == "ionq_qpu":
            job = backend.run(optimized_laplacian, shots=shots)
            import time
            while job.status() is not JobStatus.DONE:
                print("Job status is", job.status())
                time.sleep(60)
            result = job # sic, weird IonQ convention

            if rename_job_after_done:
                print("Warning: Ignoring rename job on IonQ server, no functionality.")
                rename_job_after_done = False
            rename_job(job)
        elif backend_name == "HQS-LT-S1-SIM" or backend_name == "HQS-LT-S2-SIM" \
             or backend_name == "HQS-LT-S1" or backend_name == "HQS-LT-S2":
            job_honey = execute(optimized_laplacian, backend, shots=shots)
            job_honey_id = job_honey.job_ids()[0]
            print("Honeywell Job id: ", job_honey_id)
            return (backend_name, job_honey_id)
        else:
            if job_name is None:
                job_name = str(num_vertices) + "_" \
                    + ("True" if mid_circuit else "False") + "_" \
                    + ("True" if sqrt else "False")
            job = execute(optimized_laplacian, backend, job_name=job_name,
                          shots=shots)
            # time.sleep(5)
            if job.status() is JobStatus.DONE or backend_local_sim:
                result = job.result()
                rename_job(job)
            else:
                print("Job submitted, please rerun to collect results.")
                return (backend_name, job_name)
    else:
        print("Laplacian Original:")
        print("width: ", laplacian.width())
        # print("depth: ", laplacian.depth()) -> zero if reached

    if backend_name == 'statevector_simulator':

        circ_output = result.get_statevector(optimized_laplacian)
        output_tensor = reshape_into_tensor_of_dim_main_ancilla_proj_split(circ_output,
                                                                           laplacian)

        main_qreg_select = slice(0, None) if sqrt else 0
        advanced_index_array = advanced_index_selecting_diagonal_of_split_k(laplacian)

        expectation_laplacian = output_tensor[
            main_qreg_select,
            0,
            0,
            advanced_index_array]

        if sqrt:
            expectation_laplacian = np.linalg.norm(expectation_laplacian, axis=0)**2

    else: # 'qasm_simulator' or hardware

        counts = result.get_counts()

        counts_file = open("../output/" + (backend_name + "_" if backend_any_asked else "") + job_name +
                           ".json", "w")
        json.dump(counts, counts_file)
        counts_file.close()

        frac_counts = np.array(get_fraction_count_for_all_k(counts, laplacian, sqrt))
        expectation_laplacian = frac_counts if sqrt else np.sqrt(frac_counts)

    # num_Bs = power*(1 if sqrt else 2) not actual B's, but analytical number
    # num_Bs = power*2
    #expectation_laplacian *= np.sqrt(num_vertices)**(num_Bs)  # final correction since B = sqrt(n)RXR
    
    # expectation_laplacian *= num_vertices**power times (num_vertices^power) to correct for B, divide by to make largest eigenvalue 1
  #  expectation_laplacian[0] -= 1/(num_vertices**power) # correction for augmented Laplacian, see Goldeberg Page 42  v^T(P_kP_GBP_GBP_GP_k)v, since v^TUv = 0 for Hadamard vectors, we ignore this term

    return expectation_laplacian

    # if power > 0:
    #     # expectation_projections = fraction_in_complex_of_all_orders(num_vertices=num_vertices,
    #     #                                                             edges_in=edges_in,
    #     #                                                             vec_in_circ=vec_circ)

    #     # # Boundary squared = identity * num_main_qubits
    #     # expectation_boundary_squared = expectation_projections*num_main_qubits

    #     # expectation_laplacian = ((-1)**sqrt)*((expectation_projections -
    #     #                                      expectation_taylor)/(laplacian._time_evolution**2) -
    #     #                                      expectation_boundary_squared*power)

    #     return expectation_laplacian
    # else:
    #     return expectation_taylor


def fraction_in_complex_of_all_orders(*, num_vertices, edges_in, vec_in=None, vec_in_circ=None, mid_circuit=False, sqrt=True, job_name=None, backend_name="statevector_simulator"):

    return expectation_of_laplacian(num_vertices=num_vertices, edges_in=edges_in, vec_in=vec_in,
                                    vec_in_circ=vec_in_circ, power=0, mid_circuit=mid_circuit,
                                    sqrt=sqrt, job_name=job_name, backend_name=backend_name)

# def average_expectation_over_random_two_local_vec_circs(num_vertices, edges_in, power=1, num_samples=1):

#     expectation_val = 0
#     for i in range(num_samples):

#         random_circ = TwoLocal(num_vertices, 'ry', 'cx', 'linear',
#                                parameter_prefix='theta', reps=1)

#         param_dict = {}
#         for param in random_circ.ordered_parameters:
#             param_dict[param] = np.random.rand()*2*np.pi

#         random_circ = random_circ.bind_parameters(param_dict)

#         expectation_run = expectation_of_laplacian(num_vertices=num_vertices, edges_in=edges_in,
#                                                    vec_in_circ=random_circ,
#                                                    power=power, backend_name="statevector_simulator")
#         expectation_val += expectation_run

#     expectation_val = expectation_val/num_samples

#     return expectation_val

def expectation_over_random_hadamard_vecs(num_vertices, edges_in, powers=[1], num_samples=10, noise = [0.001,0.1], expt_name= None, gpu = True):

    # expectation_val = 0
    expectation_matrix = np.zeros((num_samples,
                                  int(len(powers)),
                                   num_vertices-1))

    for sample in range(num_samples):

        random_circ = QuantumCircuit(num_vertices)
        count_h = 0
        for i in range(num_vertices):
            if (np.random.rand() < 0.5):
                count_h+=1
                random_circ.x(i)
        if count_h == 0:
            random_circ.x(np.random.randint(0,num_vertices))
            
        for i in range(num_vertices):
            random_circ.h(i)

        for power_index, power in enumerate(powers):
            
            if expt_name is not None:
                job_name = expt_name+"degree"+str(power_index)+"sample"+str(sample)
                if gpu:
                    b_name = "qasm_simulator"
                else:
                    b_name = "ibmq_qasm_simulator"    
                temp = expectation_of_laplacian(num_vertices=num_vertices,
                                         edges_in=edges_in,
                                         vec_in_circ=random_circ,
                                         power=power,
                                         mid_circuit=True,
                                         backend_name=b_name, noise = noise, job_name = job_name, gpu = gpu)
                if isinstance(temp[0], str):
                    continue
                else:
                    expectation_matrix[sample, power_index, :] =  temp 
            else:
                expectation_matrix[sample, power_index, :] = \
                    expectation_of_laplacian(num_vertices=num_vertices,
                                         edges_in=edges_in,
                                         vec_in_circ=random_circ,
                                         power=power,
                                         mid_circuit=True,
                                         backend_name="qasm_simulator", noise = noise)
                
    return expectation_matrix


def help_setting_laplacian_arguments(num_vertices=4, mid_circuit=True, do_rccx=True, sqrt=True, num_resets=None, backend_name=None, make_edges_in=10, make_vec_in_circ=5, noise_lvl=None, shots=1000):

    if make_edges_in == 0:
        edges_in, min_num_vert, edges_in_type = [], 2, "noedge"
    elif  make_edges_in == 1:
        edges_in, min_num_vert, edges_in_type = [[0,1]], 2, "edge"
    elif make_edges_in == 2:
        edges_in, min_num_vert = one_skeleton_triangle()
        edges_in_type = "triangle"
    elif make_edges_in == 3:
        edges_in, min_num_vert = one_skeleton_square()
        edges_in_type = "square"
    elif make_edges_in == 4:
        edges_in, min_num_vert = one_skeleton_tetrahedron()
        edges_in_type = "tetrahedron"
    elif make_edges_in == 5:
        edges_in, min_num_vert = one_skeleton_square_with_diagonal()
        edges_in_type = "square_diag"
    elif make_edges_in == 6:
        edges_in, min_num_vert = one_skeleton_pyramid()
        edges_in_type = "pyramid"
    elif make_edges_in == 7:
        edges_in, min_num_vert = one_skeleton_unfilled_cube()
        edges_in_type = "cube"
    elif make_edges_in == 8:
        edges_in, min_num_vert = one_skeleton_n_disconnected_squares(num_vertices)
        edges_in_type = "disconnected_squares"
    elif make_edges_in == 9:
        one_skel, _ = one_skeleton_fully_connected(num_vertices)
        edges_in = one_skel[:len(one_skel)//2]
        min_num_vert = num_vertices
        edges_in_type = "half_all_edges"
    elif make_edges_in == 10:
        one_skel, _ = one_skeleton_fully_connected(num_vertices//2)
        edges_in = one_skel + (np.array(one_skel) + [[num_vertices//2, num_vertices//2]]).tolist()
        min_num_vert = num_vertices
        edges_in_type = "two_fully_connected_clusters"
    else:
        raise ValueError("Unknown option for 'make_edges_in': ", make_edges_in)

    if min_num_vert > num_vertices:
        raise ValueError("Need more vertices than requested: " +
                         str(min_num_vert) + " > " + str(num_vertices))

    if make_vec_in_circ == 1:
        vec_in_circ = QuantumCircuit(num_vertices)
        for i in range(num_vertices):
            if (np.random.rand() < 0.5):
                vec_in_circ.x(i)
        for i in range(num_vertices):
            vec_in_circ.h(i)
        print(vec_in_circ._data)
        vec_in_circ_type = "random_vec"
    elif make_vec_in_circ == 2:
        vec_in_circ = QuantumCircuit(num_vertices)
        vec_in_circ.x(0)
        vec_in_circ.x(1)
        vec_in_circ_type = "edge_only_vec"
    elif make_vec_in_circ == 3:
        vec_in_circ = None
        vec_in_circ_type = "uniform_vec"
    elif make_vec_in_circ == 4:
        vec_in_circ = QuantumCircuit(num_vertices)
        vec_in_circ.x(0)
        vec_in_circ.h(1)
        vec_in_circ.cx(1, 0)
        vec_in_circ_type = "two_vertices_vec"
    elif make_vec_in_circ == 5:
        vec_in_circ = QuantumCircuit(num_vertices)
        vec_in_circ.x(0)
        for i in range(num_vertices):
            vec_in_circ.h(i)
        vec_in_circ_type = "fixed_rademacher"
    else:
        raise ValueError("Unknown option for 'make_vec_in_circ': ", make_vec_in_circ)

    if noise_lvl == 0:
        noise = None                
    elif noise_lvl == 1:
        noise = [0.0001, 0.001]
    elif noise_lvl == 2:
        noise = [0.001, 0.01]
    elif noise_lvl == 3:
        noise = [0.01, 0.1]
    elif noise_lvl == 4:
        noise = [10**(-5), 10**(-4)]
    elif noise_lvl == 5:
        noise = [10**(-4), 0.001]
    elif noise_lvl == 6:
        noise = [0.0002, 0.002]
    elif noise_lvl == 7:
        noise = [0.0005, 0.005]
    elif noise_lvl == 8:
        noise = [0.001, 0.01]
    else:
        raise ValueError("Unknown option for 'noise_lvl': ", noise_lvl)

    job_name = ((backend_name + "_" if backend_name is not None else "")
                + edges_in_type + "_"
                + "vert_" + str(num_vertices) + "_"
                + "mid_circ_" + str(mid_circuit) + "_"
                + "sqrt_" + str(sqrt) + "_"
                + (("r_" + str(num_resets) +
                    "_") if mid_circuit and num_resets is not None else "")
                + vec_in_circ_type + "_"
                + ("rccx_" if do_rccx else
                   "ccx_")
                + (str(noise[0])+"_"+str(noise[1]) + "_"
                   if noise is not None else
                   "no_noise_")
                + "shots_" + ("*" if shots is None else str(shots)))
                        
    return edges_in, min_num_vert, edges_in_type, vec_in_circ, vec_in_circ_type, noise, job_name

##########################################################################

def compare_counts_hellinger(counts1, counts2):
    if isinstance(counts1, str):
        counts1 = json.load(open(counts1, "r"))
    elif not isinstance(counts1, dict):
        raise ValueError("Argument must be filename string or counts dictionary.")

    if isinstance(counts2, str):
        counts2 = json.load(open(counts2, "r"))
    elif not isinstance(counts2, dict):
        raise ValueError("Argument must be filename string or counts dictionary.")

    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    all_keys = set(list(counts1.keys()) + list(counts2.keys()))
    hellinger = 0
    for key in all_keys:
        hellinger += (np.sqrt(counts1.get(key, 0)/total1) - np.sqrt(counts2.get(key, 0)/total2))**2

    hellinger /= 2

    return hellinger

def compare_counts(counts1, counts2, interpret_counts_as_integers=False, wrap_at=0, kl_frac=True):

    if isinstance(counts1, str):
        counts1 = json.load(open(counts1, "r"))
    elif not isinstance(counts1, dict):
        raise ValueError("Argument must be filename string or counts dictionary.")

    if isinstance(counts2, str):
        counts2 = json.load(open(counts2, "r"))
    elif not isinstance(counts2, dict):
        raise ValueError("Argument must be filename string or counts dictionary.")

    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    matching_counts = []

    if interpret_counts_as_integers:

        counts1_list_of_reg_strings = list(map(lambda count: count.split(), counts1.keys()))
        counts1keys = list(map(bin_string2int, counts1_list_of_reg_strings))

        counts1values = list(counts1.values())

        counts2_list_of_reg_strings = list(map(lambda count: count.split(), counts2.keys()))
        counts2keys = np.array(list(map(bin_string2int, counts2_list_of_reg_strings)))

        # if bug_fix_ancilla:
        #     counts2keys = np.delete(counts2keys, -2, axis=1)
        # # uneccessarily long for noise, but saves me from calc lookup

        num_vertices_2 = len(counts2_list_of_reg_strings[0][-1])
        counts2values = list(counts2.values())

        for place_in_counts1, integer_list_key_in_counts1 in enumerate(counts1keys):

            if wrap_at:
                source_keys_count_reg = (integer_list_key_in_counts1[0] +
                    np.arange(0, num_vertices_2, wrap_at)).reshape(-1,1)
                num_source = source_keys_count_reg.shape[0]
                remaining_regs = integer_list_key_in_counts1[1:]
                num_remaining = len(remaining_regs)
                group_keys = np.concatenate((source_keys_count_reg,
                                             np.broadcast_to(remaining_regs,
                                                             (num_source, num_remaining))),
                                            axis=1).tolist()
            else:
                group_keys = [integer_list_key_in_counts1]

            accumulate_counts = 0

            for integer_list_key_in_group in group_keys:
                look_for_key = np.all(counts2keys == integer_list_key_in_group, axis=1)
                place_in_counts2 = np.where(look_for_key)[0]
                if place_in_counts2.size > 0:
                    accumulate_counts += counts2values[place_in_counts2[0]]

            if accumulate_counts == 0:
                print("KL: to infinity and beyond!")
                return np.inf

            matching_counts.append((counts1values[place_in_counts1],
                                    accumulate_counts))
    else:
        for i in counts1.keys():
            count2 = counts2.get(i)
            if count2 is None:
                print("KL: to infinity and beyond!")
                return np.inf
            matching_counts.append((counts1[i], count2))

    matching_counts = np.array(matching_counts)
    kl_div = sum(rel_entr(matching_counts[:, 0]/total1, matching_counts[:, 1]/total2))

    if kl_frac:
        num_qubits = sum(map(len, next(iter(counts2)).split()))
        entropy_counts1 = sum(entr(matching_counts[:, 0]/total1))
        kl_div = kl_div/(num_qubits*np.log(2)-entropy_counts1)
        print("KL Fraction: ", kl_div)
        return kl_div
    else:
        print("KL: ", kl_div)
        return kl_div

def merge_counts(dict_or_filenames_list, clip_val=None, keep_keys=None, template=None, salt_rest=False):
    merged_dict = {}
    if len(dict_or_filenames_list) == 0:
        ValueError("Argument must be filename string or counts dictionary list.")
    for counts_dict in dict_or_filenames_list:
        if isinstance(counts_dict, str):
            counts_dict = json.load(open(counts_dict, "r"))
        elif not isinstance(counts_dict, dict):
            raise ValueError("Argument must be filename string or counts dictionary list.")

        if template is not None:
            template = template[::-1]
            fixed_counts_dict = {}
            num_registers = len(template)
            expected_key_len = sum(template)
            for key in counts_dict:
                # if key =='1':
                # pdb.set_trace()
                actual_key_len = len(key)
                padded_key = '0'*(expected_key_len - actual_key_len) + key
                start_index = 0
                end_index = template[0]
                # Reverse
                # start_index = -template[0]
                # end_index = expected_key_len
                new_key = ''
                for i in range(num_registers):
                    new_key = padded_key[start_index:end_index] + new_key
                    if i < num_registers-1:
                        new_key = ' ' + new_key
                        start_index = end_index
                        end_index = end_index + template[i+1]
                        # Reverse
                        # end_index = start_index
                        # start_index = start_index - template[i+1]
                    
                fixed_counts_dict[new_key] = counts_dict[key]
            counts_dict = fixed_counts_dict

        for key in counts_dict: # actually don't need to run if only one dict
            old_val = merged_dict.get(key, 0)
            merged_dict[key] = old_val + counts_dict[key]

    normalize_count = sum(list(merged_dict.values()))

    if keep_keys is not None:
        new_dict = {}
        for key in keep_keys:
            old_val = merged_dict.get(key, 0)
            if old_val > 0:
                new_dict[key] = old_val
    else:
        new_dict = merged_dict

    if clip_val is not None:
        new_dict_2 = {}
        # normalize_count = 0
        for key in new_dict:
            old_val = new_dict.get(key)
            # normalize_count += old_val
            if old_val > clip_val:
                new_dict_2[key] = old_val
                # del merged_dict[key] -> problem in for loop
        new_dict = new_dict_2

    if (keep_keys or clip_val):
        new_dict['rest' + (secrets.token_urlsafe(8) if salt_rest else "")] = normalize_count-sum(new_dict.values())
        #     new_dict[key] = int((new_dict.get(key)/normalize_count)*10**6)
        # new_dict['rest'] = int(10**6-sum(new_dict.values()))

    return new_dict

def load_merge_compare(list1, list2, interpret_counts_as_integers=True, wrap_at=0, kl_frac=True):
    list1_counts = merge_counts(list1)
    list1_counts_num_shots = sum(list1_counts.values())

    list2_counts = merge_counts(list2)
    list2_counts_num_shots = sum(list2_counts.values())

    print("List 1 counts:", list1_counts_num_shots)
    print("List 2 counts:", list2_counts_num_shots)

    kl = compare_counts(list1_counts, list2_counts,
                        interpret_counts_as_integers=interpret_counts_as_integers,
                        wrap_at=wrap_at, kl_frac=kl_frac)

    return kl

def expectation_of_laplacian_json(filename, num_vertices, power=1, mid_circuit=True, sqrt=True, num_resets=None,
do_rccx=True ): 
    
    laplacian = Projected_Laplacian(num_vertices=num_vertices, edges_in=[], power=power,
                                    mid_circuit=mid_circuit, sqrt=sqrt, num_resets=num_resets,
                                    do_rccx=do_rccx)
    template = [laplacian._count_bits_per_projection]*laplacian._num_split_projections + [laplacian._num_pairs]*laplacian._num_complex_projections + [num_vertices]

    counts = merge_counts([filename], clip_val=None, keep_keys=None, template=template)
    
    frac_counts = np.array(get_fraction_count_for_all_k(counts, laplacian, sqrt))
    expectation_laplacian = frac_counts if sqrt else np.sqrt(frac_counts)
    
    return expectation_laplacian

def calc_circuit_depths(*, log_num_vertices_start=1, log_num_vertices_end=6, power=1, mid_circuit=False, sqrt=True, do_rccx=True, num_resets=None, backend_name_or_Fake="statevector_simulator"):
 
    # ibmq_qasm_simulator ibmq_armonk ibmq_montreal ibmq_toronto ibmq_santiago ibmq_bogota ibmq_manhattan ibmq_casablanca ibmq_sydney ibmq_mumbai ibmq_lima ibmq_belem ibmq_quito ibmq_guadalupe ibmq_brooklyn ibmq_jakarta ibmq_manila ibm_hanoi ibm_lagos ibm_cairo
# https://github.com/Qiskit/qiskit-terra/tree/795e845bd83deda54c527bb3d750df00d9d03431/qiskit/test/mock/backends

    # properties = backend.properties()
    # coupling_map = backend.configuration().coupling_map
    # simulator = Aer.get_backend('qasm_simulator')

    depths = []
    backend = None
    honeywell=False

    if backend_name_or_Fake == "ionq_simulator":
        from qiskit_ionq import IonQProvider
        provider = IonQProvider()
        # backend = provider.get_backend("ionq_simulator")
        # provider = IonQProvider("YKI565sxYHK1Pe3w4mRRHOQ6EvhbGXeF")
        # print(provider.backends())
        # backend = provider.get_backend("ionq_qpu")
    elif backend_name_or_Fake in ['HQS-LT-S2-APIVAL', 'HQS-LT-S2-SIM', 'HQS-LT-S1-SIM', 'HQS-LT-S1-APIVAL', 'H1-2E',
                                  'H1-1E']:
        Honeywell.load_account()
        # backends = Honeywell.backends()
        backend = Honeywell.get_backend(backend_name_or_Fake)
        honeywell=True
    elif backend_name_or_Fake == "statevector_simulator" or \
       backend_name_or_Fake == "qasm_simulator":
        backend = BasicAer.get_backend(backend_name_or_Fake)
    elif backend_name_or_Fake.__class__.__bases__[0] is fake_pulse_backend.FakePulseBackend:
        backend = backend_name_or_Fake
    else:
        raise(ValueError("Error: backend_name_or_Fake not sim string or FakeBackend."))

        # exec("from qiskit.test.mock import " + backend_name)
        # # Security risk!
        # exec("backend = " + backend_name + "()")
        # pass # Wow, weird error if leave out

    for log_num in range(log_num_vertices_start, log_num_vertices_end):
        num_vertices = 1 << log_num
        vec_circ = QuantumCircuit(num_vertices)
        for i in range(num_vertices//2):
            vec_circ.x(i)
        for i in range(num_vertices):
            vec_circ.h(i)

        vec_circ_inst = vec_circ.to_instruction()
        vec_circ_inverse_inst = \
            vec_circ.copy().inverse().to_instruction()

        one_skel, _ = one_skeleton_fully_connected(num_vertices)
        half_one_skel = one_skel[:len(one_skel)//2]
        laplacian = Projected_Laplacian(num_vertices=num_vertices, edges_in=half_one_skel, power=power,
                                        mid_circuit=mid_circuit, sqrt=sqrt, num_resets=num_resets,
                                        do_rccx=do_rccx)
        q_main_reg = laplacian._main_qreg
        num_main_qubits = laplacian._num_main_qubits
        num_pairs = laplacian._num_pairs
        num_total_qubits = laplacian._num_total_qubits

         # CIRCUIT CONSTRUCTION
        laplacian.append(vec_circ_inst, qargs=q_main_reg)

        laplacian.power(power) # already taking into account sqrt (during _init_)

        if not sqrt: # here and below could use laplacian._sqrt to emphasise laplacian, chose not to since did assign local variables to other laplcian variables
            laplacian.append(vec_circ_inverse_inst, qargs=q_main_reg)

            # print_unitary_and_trace(laplacian)

        if backend_name_or_Fake != 'statevector_simulator': #may need to check for ibmq statevector_simulator
            laplacian.measure_all_qregs()

        unrolled_laplacian = transpile(laplacian, backend=backend,
                                seed_transpiler=11,
                                optimization_level=0)
        print(unrolled_laplacian.count_ops())
        unrolled_depth = unrolled_laplacian.depth()
        print("width: ", unrolled_laplacian.width(), "depth: ", unrolled_depth)

        print("Laplacian Optimized:")
        optimized_laplacian = transpile(laplacian, backend=backend,
                                        seed_transpiler=11, optimization_level=3)
        print(optimized_laplacian.count_ops())
        optimized_depth = optimized_laplacian.depth()
        print("width: ", optimized_laplacian.width(), "depth: ", optimized_depth)
        
        # if honeywell:
        #     job = execute(unrolled_laplacian, backend)
        #     print("HQC: ", job.__dict__)#job._experiment_results)#[0]['cost'])

        depths.append((num_vertices, unrolled_depth, optimized_depth))

    return depths


def produce_depths_plot(do_rccx=True):
    import matplotlib.pyplot as plot

    # depths_False_True = np.array(calc_circuit_depths(mid_circuit=False,
    #                                                  sqrt=True, do_rccx=do_rccx))

    depths_True_True = np.array(calc_circuit_depths(log_num_vertices_end=8, mid_circuit=True,
                                                    sqrt=True, do_rccx=do_rccx, power=3,
                                                    backend_name_or_Fake="qasm_simulator"))

    # depths_True_True_Honeywell2 = np.array(calc_circuit_depths(log_num_vertices_end=4, mid_circuit=True,
    #                                                 sqrt=True, do_rccx=do_rccx,
    #                                                 backend_name_or_Fake='H1-2E'))
    depths_True_True_Honeywell1 = np.array(calc_circuit_depths(log_num_vertices_end=4, mid_circuit=True,
                                                    sqrt=True, do_rccx=do_rccx, power=3,
                                                    backend_name_or_Fake='H1-1E'))
                                                    # backend_name_or_Fake="HQS-LT-S1-APIVAL"))

    # from qiskit.test.mock import FakeManhattan
    # fake_backend = FakeManhattan()
    # depths_True_True_fake = \
    #     np.array(calc_circuit_depths(mid_circuit=True,
    #                                  sqrt=True, do_rccx=do_rccx,
    #                                  backend_name_or_Fake=fake_backend))

    # depths_True_True_IonQ = np.array(calc_circuit_depths(mid_circuit=True,
    #                                                 sqrt=True, do_rccx=do_rccx,
    #                                                 backend_name_or_Fake="ionq_simulator"))

    # mycolors = ('green', 'blue', 'orange', 'red')
    # mycirc_names = ('Statevector Simulation', 'Mid-circuit QASM all-to-all connectivity', 'Mid-circuit IonQ all-to-all connectivity', 'Mid-circuit IBM hardware connectivity')
    # circuits = [depths_False_True, depths_True_True, depths_True_True_IonQ,
    #                depths_True_True_fake]

    # mycolors = ('blue', 'magenta')
    mycolors = ("#0c276c", "#7b1547")
    mydarkcolors = ("#648fff", "#dc006c")
    mycirc_names = ('QASM all-to-all connectivity', 'Ion-trap')
    circuits = [depths_True_True, depths_True_True_Honeywell1]
    
    fig, ax = plot.subplots()
    ax.set_title("Depth vs Num Vertices")
    ax.set_xlabel("Number of Vertices")
    ax.set_ylabel("Depth")
    for i, depths in enumerate(circuits):
        print("Depths: ", depths)
        ax.plot(depths[:, 0], depths[:, 1], label="Non-optimized " + mycirc_names[i], color=mycolors[i])
        ax.plot(depths[:, 0], depths[:, 2], label="Optimized " + mycirc_names[i],
                  color=mydarkcolors[i])#'dark'+mycolors[i])

    ax.legend()
    plot.show()
    # plot.savefig("../output/circuit_depth_vs_vertices_with_honeywell" + ("rccx" if do_rccx else "ccx") + ".png")


def honeywell_save_account_and_info():
    Honeywell.save_account('lhoresh@us.ibm.com')
    Honeywell.load_account()
    # backend = Honeywell.get_backend('HQS-LT-S1-APIVAL')
    # backend = Honeywell.get_backend('HQS-LT-S2-APIVAL')
    backends = Honeywell.backends()
    print("Backends: ", backends)
        # for backend in honey_backends:
    #     # backend = Honeywell.get_backend(backend_name)
    #     print(backend._configuration.n_qubits)
    # raise(ValueError("Stop, but keep interpreter."))
   #  backend = Honeywell.get_backend('HQS-LT-S1-APIVAL')
   #  backend = Honeywell.get_backend('HQS-LT-S2-APIVAL')
 
def honeywell_retrieve_counts(backend_name, job_honey_id):
    Honeywell.load_account()
    backend = Honeywell.get_backend(backend_name)
    job = backend.retrieve_job(job_honey_id)
    result = job.result()
    counts = result.get_counts()
    print(counts)
    json.dump(counts, open('../output/' + job_honey_id + '.json', 'w'))


edge_2_qasm = \
    "../output/qasm_simulator_edge_2_True_True_r_None_uniform_vec_rccx_*.json"
# wQwZaMMxCsc.json"

square_4_qasm = \
    "../output/qasm_simulator_square_4_True_True_r_None_uniform_vec_rccx_*.json"
# Oo7V2RA5lHI.json"

cube_8_qasm = \
    "../output/qasm_simulator_cube_8_True_True_r_None_uniform_vec_rccx_2Wh8W7eFp9k.json"
#_*.json" #DB7w1fJK6e4.json"

edge_2_honeywell = \
        "../output/32f5839d79674998aa80c261dfe8a1bb.json"
square_4_honeywell = \
        "../output/c90c48da23c448dc97f6054bd9d45cee.json"
cube_8_honeywell = \
        "../output/5b5a56527ed24073aab10011699e493d.json"

if __name__ == '__main__':

    
    # depths = calc_circuit_depths(log_num_vertices_start=3, log_num_vertices_end=3, power=1, mid_circuit=True, backend_name_or_Fake='qasm_simulator')
    # print("Depths: ", depths)
    # raise(ValueError("Stop, but keep interpreter."))
    
   # depths = calc_circuit_depths(log_num_vertices_end=4, mid_circuit=True, backend_name_or_Fake='HQS-LT-S1-APIVAL')
   # print("Depths: ", depths)
   # raise(ValueError("Stop, but keep interpreter."))

    #####################################
    # Depth plots
    #####################################
    honeywell_save_account_and_info()
    produce_depths_plot(do_rccx=True)
    # raise ValueError("Exiting, but keeping interpreter.")

    #####################################
    # Unit test all
    #####################################

    # import unittest
    # unittest.main(module='test.test_quantum_homology_advanced')

    #####################################
    # Unit test specific
    #####################################

#    import test.test_quantum_homology_advanced
#    mylaplace = test.test_quantum_homology_advanced.Expectation_of_Laplacian_TestCase()
#    mylaplace.setUp()
    # mylaplace.test_expectation_01()  # (backend="ionq_qpu")
    # mylaplace.test_expectation_02()  # (backend="ionq_qpu")
    # mylaplace.test_expectation_03()  # (backend="ionq_qpu")
    # mylaplace.test_expectation_04()  # (backend="ionq_qpu")
    # mylaplace.test_expectation_05()  # (backend="ionq_qpu")
    # # mylaplace.test_expectation_05()
#    mylaplace.test_run_random()
#    raise(ValueError("Stop, but keep interpreter."))

    
    # honeywell_save_account_and_info()
    # honeywell_retrieve_counts('HQS-LT-S2', 'c90c48da23c448dc97f6054bd9d45cee')
    # honeywell_retrieve_counts('HQS-LT-S2', '5b5a56527ed24073aab10011699e493d')
    
    pass
# job.result()
# Result(backend_name='ionq_qpu', backend_version='0.0.1', qobj_id='None', job_id='5286dc3f-fdbd-4a6b-8231-af79375e143d', success=True, results=[ExperimentResult(shots=100, success=True, meas_level=2, data=ExperimentResultData(counts={'0x1': 1, '0x2': 4, '0x4': 2, '0x5': 44, '0x6': 38, '0x7': 1, '0xd': 4, '0xe': 1, '0xf': 2, '0x14': 1, '0x16': 2}, probabilities={'0x1': 0.01, '0x2': 0.04, '0x4': 0.02, '0x5': 0.44, '0x6': 0.38, '0x7': 0.01, '0xd': 0.04, '0xe': 0.01, '0xf': 0.02, '0x14': 0.01, '0x16': 0.02}), header=QobjExperimentHeader(memory_slots=5, global_phase=2.356194490192344, n_qubits=6, name='circuit-16', creg_sizes=[['c0', 2], ['power1complexcproj0', 1], ['power1complexcproj1', 1], ['power1splitcproj0', 1]], clbit_labels=[['c0', 0], ['c0', 1], ['power1complexcproj0', 0], ['power1complexcproj1', 0], ['power1splitcproj0', 0]], qreg_sizes=[['mainqreg', 2], ['clean_ancillae', 1], ['power1complexqproj0', 1], ['power1complexqproj1', 1], ['power1splitqproj0', 1]], qubit_labels=[['mainqreg', 0], ['mainqreg', 1], ['clean_ancillae', 0], ['power1complexqproj0', 0], ['power1complexqproj1', 0], ['power1splitqproj0', 0]]))], time_taken=2.579)
