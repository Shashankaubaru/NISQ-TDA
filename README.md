# NISQ-TDA
Qiskit (python) code for Topological data analysis on noisy quantum computers

See example_notebook for sample runs of the algorithm

Datasets: 

          (a) simple geometric structure datasets are in classical_homology.py

          (b) CMB dataset is generated in cmb_sample_sky.py
          
Main files: 

          (a) quantum_homology - full NISQ-TDA agorithm with QISKIT circuit compilation. Using aer.backend, choose the machine to run

          (b) quantum_homology_sim - full NISQ-TDA agorithm with QISKIT circuit compilation and simuation
            
          (c) quantum_rank_estimation - new quantum algorithm for matrix rank estimation
            
          (d) quantum_gpu - quantum simulations with GPU enabled
            
 Tested on: 
 
          (a) Qiskit version 0.42
          (b) Qiskit Aer simulator (Aer.get_backend('aer_simulator') and 'qasm_simulator')
          (c) IBMQ machine : (provider = IBMQ.get_provider(hub='ibm-q-internal') , provider.get_backend(backend_name))
          (d) Honeywell Quantinuum machine: (Honeywell.get_backend('HQS-LT-S1') and 'HQS-LT-S1-SIM')
 
Contact: Shashanka Ubaru - shashanka.ubaru@ibm.com
