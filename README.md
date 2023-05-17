# NISQ-TDA
Qiskit (python) code for Topological data analysis on noisy quantum computers

See example_notebook for sample runs of the algorithm

Datasets: 

          (a) simple geometric structure datasets are in classical_homology.py

          (b) CMB dataset is generated in cmb_sample_sky.py
          
Main files: 

            (a) quantum_homology - full NSQ-TDA agorithm with QISKIT circuit compilation. Using aer.backend, choose the machine to run

            (b) quantum_homology_sim - full NSQ-TDA agorithm with QISKIT circuit compilation and simuation
            
            (c) quantum_rank_estimation - new quanutm algorithm for matrix rank estimation
            
            (d) quantum_gpu - quantum simulations with GPU enabled
 
Contact: Shashanka Ubaru - shashanka.ubaru@ibm.com
