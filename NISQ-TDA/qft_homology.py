# from qiskit.circuit import Parameter

#For state vector initializing
# from qiskit.aqua.components.initial_states.custom import Custom
# from qiskit.quantum_info import Pauli
# from qiskit.aqua.operators import MatrixOperator, WeightedPauliOperator, op_converter
# from qiskit.aqua.utils import decimal_to_binary, CircuitFactory, get_subsystem_density_matrix

# from qiskit.aqua.algorithms import ExactEigensolver
# from qiskit.aqua.components.iqfts import Standard as StandardIQFTS
# from qiskit.aqua.algorithms import QPE
# from qiskit.aqua.operators.op_converter import to_matrix_operator
# from qiskit.aqua.circuits import FourierTransformCircuits as Fourier_circ
# from qiskit.circuit.quantumregister import Qubit
# #from qiskit.circuit import Instruction
# from qiskit.extensions.quantum_initializer.initializer import Initialize

# #For running
# from qiskit import BasicAer
# from qiskit.aqua import QuantumInstance
# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
# from qiskit.aqua.utils import summarize_circuits
# iqft = QFT(num_ancillae).inverse()

# my_phase_estimation_circuit = PhaseEstimationCircuit(iqft=iqft, num_ancillae=num_ancillae, unitary_circuit_factory=test_boundary_factory)

# mycirc += my_phase_estimation_circuit.construct_circuit(ancillary_register=myancillae)

# # mycirc.measure(myancillae, c_ancilla)

# simulator = 'qasm_simulator'
# # simulator = 'statevector_simulator'
# backend = BasicAer.get_backend(simulator)

# result = execute(mycirc, backend=backend).result()
# ancilla_counts = result.get_counts(mycirc)
# post_select_ancilla_counts = get_subsystems_counts(ancilla_counts,post_select_index=1,post_select_flag='0'*test_boundary_factory._num_projection_bits)
# top_measurement_label = \
#          sorted([(post_select_ancilla_counts[k], k) for k in post_select_ancilla_counts])[::-1][0][-1][::-1]

# _binary_fractions = [1 / 2 ** p for p in range(1, num_ancillae + 1)]

# top_measurement_decimal = sum(
#      [t[0] * t[1] for t in zip(_binary_fractions,
#                                [int(n) for n in top_measurement_label])]
#  )

# print(top_measurement_label)
# print(top_measurement_decimal)

# mycirc.count_ops()
# summarize_circuits(mycirc)
# summarize_circuits(mycirc.decompose())

#mycirc.draw()

# ########################################
# ### CALCULATE THE NUMBER OF SIMPLICES
# ########################################
# #The quantum circuit to arrive at num_complex_simplices e.g = [3,3,0]
# #This can possibly be dequantized! Ken!? -> Yes, according to Ken

# num_complex_simplices = np.zeros(num_qubits)
# num_complex_simplices[0] = num_qubits

# mycirc = QuantumCircuit(mycomplex, mycounter)
# #create_unfilled_triangle_circ(mycirc, mycomplex)
# mycirc.initialize(state_vector, mycomplex)
# split_k(mycirc, mycomplex, mycounter)

# simulator = 'statevector_simulator'
# backend = BasicAer.get_backend(simulator)
# result = execute(mycirc, backend=backend).result()
# mystate=result.get_statevector(mycirc).reshape(num_basis, num_counts, order='F')

# projected_state = mystate[:, 1]
# total_num_simplices = num_qubits/norm(projected_state)**2

# for k in range(1, num_qubits):
#     projected_state = mystate[:, k+1]
#     print(projected_state)
#     num_complex_simplices[k] = (norm(projected_state)**2)*total_num_simplices

# print("Number of complex simplices of each degree:")
# print(num_complex_simplices)

# ########################################

# mycirc.count_ops()
# summarize_circuits(mycirc.decompose())
# mycirc.draw()

########################################
### Construct Vitoris Vector
########################################

# vit_vec = construct_unfilled_triangle()
# vit_vec = construct_unfilled_square()
# vit_vec_no_signs = np.abs(vit_vec)

# num_qubits = num_underlying_vertices_vector_length(len(vit_vec_no_signs))

# num_basis = 1 << num_qubits

# state_vector = 1/norm(vit_vec_no_signs)*vit_vec_no_signs

# print("State Vector: ", state_vector)
# print("Num qubits: ", num_qubits)


# mycomplex = QuantumRegister(num_qubits)
# #mycirc = QuantumCircuit(mycomplex, mycounter)
# mycirc = QuantumCircuit(mycomplex)

# mycirc.initialize(state_vector, mycomplex)
# simulator = 'statevector_simulator'
# backend = BasicAer.get_backend(simulator)
# result = execute(mycirc, backend=backend).result()
# backend = BasicAer.get_backend(simulator)
# result = execute(mycirc, backend=backend).result()
# mystate=result.get_statevector(mycirc).reshape(num_basis, 1, order='F')
# print_matrix(mystate)

# b = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
#full_circ.isometry(state_vector, None, simplicial_complex_register)
# temp_reg = QuantumRegister(boundary_evolution_instruction.num_qubits)
# temp_circ = QuantumCircuit(temp_reg)
# temp_circ.append(boundary_evolution_instruction, qargs=temp_reg)
# boundary_evolution_unitary = execute(temp_circ, backend=UnitarySimulatorPy()).result().get_unitary()

# num_ancillae = 5

# exact_eigensolver = NumPyMinimumEigensolver(PrimitiveOp(boundary_evolution_unitary))
# results = exact_eigensolver.run()
# ref_eigenval = results.eigenvalue
# ref_eigenvec = results.eigenstate

# ws, vs = np.linalg.eig(boundary_sum_matrix)
# ref_eigenval = ws[0]
# ref_eigenvec = vs[0]

# print('The exact eigenvalue is:       %s', ref_eigenval)
# print('The corresponding eigenvector: %s', ref_eigenvec)

# #qubit_op = boundary_evolution_operator
# qubit_op = boundary_sum

# state_in = Custom(qubit_op.num_qubits, state_vector=ref_eigenvec)
# iqft = QFT(num_ancillae).inverse()

# qpe = QPE(qubit_op, state_in, iqft, num_time_slices=1, num_ancillae=num_ancillae,
#           expansion_mode='suzuki', expansion_order=2,
#           shallow_circuit_concat=True)

# backend = BasicAer.get_backend(simulator)
# quantum_instance = QuantumInstance(backend, shots=1000, seed_transpiler=1, seed_simulator=1)

#         # run qpe
# result = qpe.run(quantum_instance)

# # report result
# print('top result str label:         %s', result.top_measurement_label)
# print('top result in decimal:        %s', result.top_measurement_decimal)
# print('stretch:                      %s', result.stretch)
# print('translation:                  %s', result.translation)
# print('final eigenvalue from QPE:    %s', result.eigenvalue)
# print('reference eigenvalue:         %s', ref_eigenval)
# print('ref eigenvalue (transformed): %s',
#                (ref_eigenval + result.translation) * result.stretch)
# print('reference binary str label:   %s', decimal_to_binary(
#     (ref_eigenval.real + result.translation) * result.stretch,
#     max_num_digits=num_ancillae + 3,
#     fractional_part_only=True
# ))


###################
#IF WE COULD CONSTRUCT EQUAL SUP OVER EIGENVECTORS, FOR EG WITH VQE MAXIMISING EXPECTATION, ALTERNATIVELY, WE HAVE THE COMPLEX, WHICH IS CLOSE TO EQUAL SUP, THIS COULD BE A GOOD LOWER BOUND!



###################

#OLD PROJECTION EXPERIMENTS

########################################
### CREATE UNFILLED TRIANGLE
########################################

# def create_unfilled_triangle_circ(circ, qreg):
#     #create full simplicial complex of unfilled triangle on first three qubits

#     circ.h(qreg[2])

#     circ.ry(1.91063323624902, qreg[1])
#     circ.ry(3*np.pi/4, qreg[0])
#     circ.cx(qreg[1], qreg[0])
#     circ.ry(np.pi/4, qreg[0])
#     circ.cx(qreg[1], qreg[0])

#     circ.cx(qreg[2], qreg[0])
#     circ.cx(qreg[2], qreg[1])

#     return circ

########################################
### PROJECTION ONTO COMPLEX
########################################

# def project_onto_triangle(circ, from_qreg, project_qreg):

#     # copy into project register (creating entangled copy)
#     circ.cx(from_qreg, project_qreg)

#     #create full simplicial complex of unfilled triangle on temp circuit
#     circ_complex = QuantumCircuit(project_qreg)
#     create_unfilled_triangle_circ(circ_complex, project_qreg)

#     #add reverse of circuit to project onto complex
#     circ += circ_complex.inverse()

# def project_onto_complex(complex_vector, circ, from_qreg, project_qreg):

#     # copy into project register (creating entangled copy)
#     circ.cx(from_qreg, project_qreg)

#     #create full simplicial complex of unfilled triangle on temp circuit
#     # circ_complex = QuantumCircuit(project_qreg)
#     # circ_complex.initialize(complex_vector, project_qreg)
#     myinit = Initialize(complex_vector)

#     # add reverse of circuit to project onto complex
#     # circ.data.append(circ_complex.data[0][0].gates_to_uncompute())
#     circ.append(myinit.gates_to_uncompute().to_instruction(), project_qreg)
#     #    circ += circ_complex.inverse()

#     #Alternative:
#     #Collapse only if boundary simplex was in original complex
#     #This is more complicated, below detects 0-1 errors
#     #still need 1-0 zero errors and then maybe X and AND and then collapse?
#     #circ.ccx(from_qreg, project_qreg, myproject)
#     #Haha, this then reminded Lior about QEEC, so next try to use QEEC for projection, does not work
#     #KEY PROBLEM IS TENSOR OF SUPERPOSITION ALWAYS CREATES ALL POSSIBILITIES
#     #same for product of coefficients, Almudena et.al!

        # simplex_qreg = QuantumRegister(num_qubits)
        # pairs_qreg = QuantumRegister(num_pairs)
        # simplex_in = QuantumRegister(num_pairs)
