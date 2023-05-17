# -*- coding: utf-8 -*-

# Copyright 2020 IBM.
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

import numpy as np
import math
from numpy import linalg as LA
from scipy.special import comb
# import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh as largest_eigsh

from quantum_homology import expectation_over_random_hadamard_vecs, expectation_of_laplacian_json

from scipy.linalg import hadamard
# Set paramters
# nv = 30
# deg = 10
# eps = 0.01
# damp = 2


def compute_moments(A, v, deg):
    Amv = np.zeros((len(v), deg+1))
    for i in range(deg+1):
        if i == 0:
            Amv[:, i] = v
        else:
            Amv[:, i] = np.dot(A, Amv[:, i-1])

    return np.dot(v, Amv)


def compute_Chebyshev_moments(x):
    deg = len(x)
    Tk = np.zeros(deg)
    for k in range(deg):
        if k == 0:
            Tk[k] = 1
        elif k == 1:
            Tk[k] = x[k]
        else:
            for j in range(int(np.floor(k/2))+1):
                akj = (comb(2*j, j)*comb(k, 2*j)) / comb(k-1, j)
                Tk[k] = Tk[k] +\
                    ((-1)**j)*((2**(k-2*j-1))*(akj*x[k-2*j]))
    return Tk


def Chebyshev_step(deg, eps, damp):
    alpha1 = eps
    alpha2 = 1
    thetJ = math.pi/(deg+2)
    thetL = math.pi/(deg+1)
    a1 = 1/(deg+2)
    a2 = math.sin(thetJ)
    beta1 = math.acos(alpha1)
    beta2 = math.acos(alpha2)
    mu = np.zeros(deg+1)
    for k in range(deg+1):
        if damp == 0:
            jac = 1
        elif damp == 1:
            jac = a1*math.sin((k+1)*thetJ)/a2 + \
                (1-(k+1)*a1)*math.cos(k*thetJ)
        elif damp == 2:
            jac = 1
            if (k > 0):
                jac = math.sin(k*thetL)/(k*thetL)

        if k == 0:
            mu[k] = -jac*(beta2-beta1)/math.pi
        else:
            mu[k] = -2*jac*(
                math.sin(k*beta2)-math.sin(k*beta1))/(math.pi*k)

    return mu



def quantum_rank_estimation(edges_in, num_ver, pwr, nv, eps, damp):

    rank = np.zeros([nv,num_ver-1])
    mu = Chebyshev_step(np.max(pwr)+1, eps, damp)
    exp_tensor = expectation_over_random_hadamard_vecs(num_ver, edges_in, powers=pwr, num_samples=nv)
    #exp_tensor = exp_value
    for l in range(nv):
        for k in range(num_ver-1):
            if np.sum(exp_tensor[l,:,k]) ==0:
                rank[l,k] = 0
                continue
            exp_value = np.append([1], exp_tensor[l,:,k])
            cheb_mom = compute_Chebyshev_moments(exp_value)
            rk = np.dot(cheb_mom, mu)
            rank[l,k] = num_ver*rk

    return np.round(np.mean(rank,0))


def quantum_Betti_estimation(edges_in, num_ver, dimHk, pwr, nv, eps, damp, noise_lvl):

    rank = np.zeros([nv,num_ver-1])
    mu = Chebyshev_step(np.max(pwr)+1, eps, damp)
    exp_tensor = expectation_over_random_hadamard_vecs(num_ver, edges_in, powers=pwr, num_samples=nv, noise_on = True, noise_lvl = noise_lvl)
    #exp_tensor = exp_value
    for l in range(nv):
        for k in range(num_ver-1):
            if np.sum(exp_tensor[l,:,k]) ==0:
                rank[l,k] = 0
                continue
            exp_value = np.append([1], exp_tensor[l,:,k])
            cheb_mom = compute_Chebyshev_moments(exp_value)
            rank[l,k] = dimHk[k]*np.dot(cheb_mom, mu)
            
    return (dimHk -  (np.mean(rank,0)))

def quantum_Betti_estimation_json(edges_in, num_ver, dimHk, pwr, nv, eps, damp):

    rank = np.zeros([nv,num_ver-1])
    mu = Chebyshev_step(np.max(pwr)+1, eps, damp)
    exp_tensor = np.zeros((nv, int(len(pwr)),num_ver-1))
    for l in range(nv):
        for power_index, power in enumerate(pwr):
            filename = "Count_square_sim_500_shots_deg"+str(power+1)+"_vec"+str(l+1)+".json"
            exp_tensor[l,power_index,:] = expectation_of_laplacian_json(filename, num_ver, power=power)

    for l in range(nv):
        for k in range(num_ver-1):
            if np.sum(exp_tensor[l,:,k]) ==0:
                rank[l,k] = 0
                continue
            exp_value = np.append([1], exp_tensor[l,:,k])
            cheb_mom = compute_Chebyshev_moments(exp_value)
            rank[l,k] = dimHk[k]*np.dot(cheb_mom, mu)
            
    return (dimHk -  (np.mean(rank,0)))
