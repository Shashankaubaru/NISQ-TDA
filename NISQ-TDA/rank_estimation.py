
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

from scipy.special import comb
from numpy import linalg as LA
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

from homology_tools import d_dim_simplices_mask, \
    num_underlying_vertices_from_vit_vec

from classical_homology import \
    classical_boundary_matrix, \
    classical_laplacian_of_vit_vec, \
    classical_projections_onto_simplices, \
    classical_boundary_sum_matrix
from scipy.linalg import hadamard
# Set paramters
nv = 30
deg = 10
eps = 0.01
damp = 2


def compute_trace_power(A, deg):
    
    trace_pow = np.zeros(deg+1)
    tmp_mat = np.eye(np.size(A,0))
    for i in range(deg+1):
        if i == 0:
            trace_pow[i] = np.trace(tmp_mat)
        else:
            tmp_mat = A@tmp_mat
            trace_pow[i] = np.trace(tmp_mat)

    return trace_pow

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

def expectation_hadamard(A, deg, nv, eps, damp):

    n = A.shape[1]
    e, vec = largest_eigsh(A, 1, which='LM')
    if np.absolute(e) <= 1e-10:
        return 0
    A = A / e
    H = hadamard(n)
#     d = (np.random.rand(n)<.5)*2 - 1;
    idx = np.random.choice(n, nv)
#     D = np.diag(d)
#     H = D @ H
    mom = np.zeros((deg+1,nv))
    for s in range(nv):
#         v = H[:,idx[s]]
        v = np.random.randn(n)
        v = v / LA.norm(v)
        mom[:,s] = compute_moments(A, v, deg)

    return mom

def rank_estimation(A, deg, nv, eps, damp):

    n = A.shape[1]
    e, vec = largest_eigsh(A, 1, which='LM')
    if np.absolute(e) <= 1e-10:
        return 0
    A = A / e
#     H = hadamard(n)
#     d = (np.random.rand(n)<.5)*2 - 1;
#     idx = np.random.choice(n, nv)
#     D = np.diag(d)
#     H = D @ H
    rank = np.zeros(nv)
    for s in range(nv):
#         v = H[:,idx[s]]
        v = np.random.randn(n)
        v = v / LA.norm(v)
        mom = compute_moments(A, v, deg)
        cheb_mom = compute_Chebyshev_moments(mom)
        mu = Chebyshev_step(deg, eps, damp)
        rk = np.dot(mu, cheb_mom)
        rank[s] = n*rk

    return np.mean(rank)



def rank_estimation_from_moments(moments, eps, damp):

    deg, nv = moments.shape
    e, vec = largest_eigsh(A, 1, which='LM')
    rank = np.zeros(nv)
    for s in range(nv):
        mom_vec = moments[:,s]
        cheb_mom = compute_Chebyshev_moments(mom_vec)
        mu = Chebyshev_step(deg, eps, damp)
        rk = np.dot(mu, cheb_mom)
        rank[s] = rk

    return np.mean(rank)


def classical_betti_from_rank_estimation_of_laplacian(vit_vec,
                                                      deg, nv,
                                                      eps, damp):

    num_vertices = num_underlying_vertices_from_vit_vec(vit_vec)

    laplacian_restricted = classical_laplacian_of_vit_vec(vit_vec)

    betti = np.zeros(num_vertices)
    for i in range(num_vertices):
        rows = d_dim_simplices_mask(i, vit_vec)
        cols = rows
        if np.any(rows) and np.any(cols):
            tmp_mat = laplacian_restricted[..., i][rows][:, cols]
            betti[i] = np.sum(rows) - (rank_estimation(
                    (tmp_mat @ tmp_mat.T), deg, nv, eps, damp))
            # np.sum(rows) equivalent to d_dim_simplices[i]
        else:
            betti[i] = 0

    return betti


def classical_betti_from_rank_estimation_of_kernel_of_boundary(
        vit_vec, deg, nv, eps, damp):

    num_vertices = num_underlying_vertices_from_vit_vec(vit_vec)
    boundary_sum_matrix = classical_boundary_sum_matrix(
        num_vertices)

    project_on_simplices = classical_projections_onto_simplices(
        vit_vec)

    kernel = np.zeros(num_vertices+1)
    kernel[0] = np.sum(d_dim_simplices_mask(0, vit_vec))

    dim_simplices = np.zeros(num_vertices+1)
    dim_simplices[0] = kernel[0]

    for i in range(num_vertices-1, 0, -1):
        i_dim_simplices = d_dim_simplices_mask(i, vit_vec)
        i_minus_1_dim_simplices = d_dim_simplices_mask(i-1,
                                                       vit_vec)
        dim_simplices[i] = np.sum(d_dim_simplices_mask(i, vit_vec))

        kernel[i] = dim_simplices[i]
        if np.any(i_minus_1_dim_simplices) and \
           np.any(i_dim_simplices):
            tmp_mat = \
                project_on_simplices[..., i-1] \
                @ boundary_sum_matrix \
                @ project_on_simplices[..., i]
            tmp_mat = \
                tmp_mat[
                    i_minus_1_dim_simplices][:, i_dim_simplices]
            kernel[i] -= (
                rank_estimation(
                    (tmp_mat.T @ tmp_mat), deg, nv, eps, damp))

    betti = np.zeros(num_vertices)
    for i in range(num_vertices):
        betti[i] = kernel[i]+kernel[i+1]-dim_simplices[i+1]
    return betti
