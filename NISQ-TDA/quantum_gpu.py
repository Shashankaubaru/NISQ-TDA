import numpy as np
import pdb
from pdb import set_trace as bp

# Import Qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.providers.aer import AerError

# Create circuit
myreg = QuantumRegister(16)
circ = QuantumCircuit(myreg)
circ.h(myreg)
circ.cx(0, 1)
circ.measure_all()


# Initialize a GPU backend
# Note that the cloud instance for tutorials does not have a GPU
# so this will raise an exception.
try:
    print(Aer.backends())
    simulator_gpu = Aer.get_backend('aer_simulator_statevector_gpu')
    # simulator_gpu.set_options(device='GPU')
    # bp()
    circ = transpile(circ, simulator_gpu)

    # Run and get counts
    result = simulator_gpu.run(circ).result()
    counts = result.get_counts(circ)
    print(len(counts))


except AerError as e:
    print(e)
