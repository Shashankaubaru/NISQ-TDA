import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
import qiskit.providers.aer.noise as qnoise

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

# Construct an ideal simulator
aersim = AerSimulator()

# Perform an ideal simulation
result_ideal = qiskit.execute(circ, aersim).result()
counts_ideal = result_ideal.get_counts(0)
print('Counts(ideal):', counts_ideal)
# Counts(ideal): {'000': 493, '111': 531}

# Construct a noisy simulator backend from an IBMQ backend
# This simulator backend will be automatically configured
# using the device configuration and noise model 
# provider = IBMQ.load_account()
# backend = provider.get_backend('ibmq_athens')
# aersim_backend = AerSimulator.from_backend(backend)
noise = [0.00001, 0.0001]
gpu = True

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

if gpu:
    # backend = Aer.get_backend('aer_simulator_statevector_gpu')
    if noise is not None:
        backend = AerSimulator(device="GPU", noise_model = noise_model, basis_gates= basis_gates,
                                 batched_shots_gpu_max_qubits = 32)
    else:
        backend = AerSimulator(device="GPU", batched_shots_gpu_max_qubits = 32)
else:
    # Perform noisy simulation

    if noise is not None:
        backend = AerSimulator(device="CPU", noise_model = noise_model, basis_gates= basis_gates)
    else:
        backend = AerSimulator(device="CPU")


result_noise = qiskit.execute(circ, backend).result()
counts_noise = result_noise.get_counts(0)

print('Counts(noise):', counts_noise)
