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

import numpy as np

# For Vitoris Rips
from scipy.spatial.distance import pdist as distance
import networkx as nx
import itertools

from homology_tools import \
    num_underlying_vertices_simplices, \
    num_underlying_vertices_vector_length, \
    num_underlying_vertices_from_vit_vec, \
    simplices_to_count_vector, \
    count_vector_to_simplices, \
    d_dim_simplices, \
    d_dim_simplices_mask

from math_fiddling import \
    print_matrix_and_info, \
    testBit, \
    clearBit, \
    condensed_to_square, \
    pair_index_to_two_power_2s, \
    two_power_2s_to_pair_index, \
    integer_to_binary_as_array, \
    is_power_of_2, \
    num_bits

B00 = np.array([[1, 0], [0, 0]])
B01 = np.array([[0, 1], [0, 0]])
Z = np.array([[1, 0], [0, -1]])
# Id = np.array([[1, 0], [0, 1]])


def B00(dim):
    B00 = np.zeros((dim, dim))
    B00[0, 0] = 1
    return B00


def classical_boundary_matrix(num_vertices):

    dim = 2**num_vertices
    boundary = np.zeros((dim, dim))

    for i in range(num_vertices):
        construct = np.array([1])
        for j in range(i):
            construct = np.kron(construct, Z)
        construct = np.kron(construct, B01)
        construct = np.kron(construct,
                            np.eye(2**(num_vertices-1-i)))
        boundary += construct

    for i in range(num_vertices):
        construct = np.kron(B00(2**i), B01)
        construct = np.kron(construct, B00(2**(num_vertices-1-i)))
        boundary -= construct

    return boundary


def classical_boundary_sum_matrix(num_vertices):
    boundary_matrix = classical_boundary_matrix(num_vertices)
    return boundary_matrix.T + boundary_matrix


def boundary_matrix_applied_to_count_vector(
        count_vector_simplices):

    len_simplices_vector = len(count_vector_simplices)

    if not is_power_of_2(len_simplices_vector):
        raise ValueError('Expected simplicial vector to be power of 2, \
        but instead: ', len_simplices_vector)

    dim = int(np.log2(len_simplices_vector))

    return (classical_boundary_matrix(dim) @ count_vector_simplices)


def boundary_map_applied_to_simplices(simplices,
                                      num_vertices=None):

    # number_of_simplices = len(simplices)
    signs = np.sign(simplices, dtype=int)
#    print(signs)
    np.abs(simplices, out=simplices, dtype=int)
#    print(simplices)

    if num_vertices is None:
        num_vertices = num_underlying_vertices_simplices(
            simplices)

    def boundary_of_one_simplex(simplex):
        boundary = []
        whichbit = 0

        # could check if simplex is power of two in order to skip single points
        # but it is still one sweep
        # consider converting to numpy operations

        for i in range(num_vertices):
            if testBit(simplex, i):
                boundary.append(((-1)**whichbit)*clearBit(simplex, i))
                whichbit += 1

        if whichbit < 2:
            boundary = []

        return boundary

    full_boundary = [thissign*np.array(boundary_of_one_simplex(insimplex), dtype=int) for thissign, insimplex in zip(signs, simplices)]

    return np.concatenate(full_boundary, axis=0)


def betti(b, simplicial_complex_count_vector):

    print("==========================================================")

    print("Starting Betti {} Calculation".format(b))

    print("Simplicial Complex: ", simplicial_complex_count_vector)

    faces = d_dim_simplices(b+1, simplicial_complex_count_vector)

    print(b+1,"-dim simplices: ", faces)

    edges = d_dim_simplices(b, simplicial_complex_count_vector)

    print(b,"-dim simplices: ", edges)

    lenvec = len(simplicial_complex_count_vector)

    num_vertices = num_underlying_vertices_vector_length(lenvec)

    #print(classical_boundary_matrix(num_vertices))

    if len(faces) != 0:
        boundary_faces = classical_boundary_matrix(num_vertices)[:, faces]
        print(b+1,"-Boundary matrix: ", boundary_faces)
        filledin = np.linalg.matrix_rank(boundary_faces)
    else:
        filledin = 0

    print("Dim of closed boundaries from higher dimensional simplices (filled in): ", filledin)

    if len(edges) != 0:
        boundary_edges = classical_boundary_matrix(num_vertices)[:, edges]
        print(b, "-Boundary of Edges: ", boundary_edges)
        #    print(boundary_edges.shape)
        allloops = boundary_edges.shape[1] - np.linalg.matrix_rank(boundary_edges)
    else:
        allloops = 0

    print("Dim of closed boundaries from cycles: ", allloops)

    print("Betti {}: {}".format(b, allloops - filledin))

    print("==========================================================")

    return allloops - filledin


def complete_unsigned_complex(count_vector_simplices):

    num_vertices = num_underlying_vertices_vector_length(
        len(count_vector_simplices))

    # nonzero_complex = \
    #     np.zeros(1 << num_vertices)
    # nonzero_complex[np.nonzero(count_vector_simplices)[0]] = 1
    k_simplices = d_dim_simplices_mask(num_vertices-1, count_vector_simplices)
    all_simplices = np.zeros((1 << num_vertices), dtype=bool)

    for i in range(num_vertices-1,-1,-1):
        all_simplices = np.logical_or(all_simplices, k_simplices)
        collect_k_minus_1 = np.zeros((1 << num_vertices), dtype=bool)
        for j in np.nonzero(k_simplices)[0]:
            dummy_complex = np.zeros((1 << num_vertices), dtype=bool)
            dummy_complex[j] = 1
            collect_k_minus_1 = np.logical_or(
                d_dim_simplices_mask(i-1,
                        boundary_matrix_applied_to_count_vector(
                        dummy_complex)),
                    collect_k_minus_1)
        if i > 0:
            k_simplices = np.logical_or(d_dim_simplices_mask(i-1, count_vector_simplices),
                            collect_k_minus_1)
        else:
            all_simplices = np.logical_or(all_simplices, collect_k_minus_1)
    return np.array(all_simplices, dtype=float)


def complex_to_pairs(complex_vector):
    num_vertices = \
        num_underlying_vertices_vector_length(len(complex_vector))

    num_pairs = int(num_vertices*(num_vertices - 1)/2)

    binary_pairs = list(map(integer_to_binary_as_array,
            np.where(d_dim_simplices_mask(1, complex_vector))[0]))
    pairs_powers = [list(np.where(np.array(pair,
                    dtype=int))[0]) for pair in binary_pairs]
    pairs_index = [two_power_2s_to_pair_index(pair[1],
                        pair[0]) for pair in pairs_powers]
    pairs = np.zeros(num_pairs, dtype=int)
    pairs[pairs_index] = 1
    # num_edges = len(pairs_index)
    # num_simplices = num_edges + num_vertices

    return pairs

def complex_one_skeleton(points, epsilon):
    number_of_points = len(points)

    #Construct graph
    condensed_distances = distance(points)
    indices = np.nonzero(condensed_distances < 2*epsilon)[0]
    distance_graph = nx.Graph()

    for k in indices:
        i, j = condensed_to_square(k, number_of_points)
        #print(i, j)
        distance_graph.add_edge(i, j)

    return [list(e) for e in distance_graph.edges]

def complex_one_skeleton_from_pairs_num_pairs_indexed(pairs):
    
    distance_graph = nx.Graph()

    for k in np.where(pairs==1)[0]:
        i, j = pair_index_to_two_power_2s(k)
        #print(i, j)
        distance_graph.add_edge(i, j)

    return [list(e) for e in distance_graph.edges]

def one_skeleton_remap(one_skeleton):
    """
    Calculates the number of needed vertices and remaps old vertices to new range (0, num_vertices).
    """
    vertices = []
    for e in one_skeleton:
        vertices.extend(e)
    old_vertices = set(vertices)
    num_vertices = len(old_vertices)
    vertex_map = {pair[0]:pair[1] for pair in zip(old_vertices, range(num_vertices))}

    return [[vertex_map[e[0]], vertex_map[e[1]]] for e in one_skeleton], 1<<num_bits(num_vertices)

# print(one_skeleton_remap([(10,20)]))

def vitoris_rips_complex_from_one_skeleton(num_vertices, edges_in):
    """ Returns VR complex as simplices from adjaceny list"""
    complex = []

    # Construct graph
    distance_graph = nx.Graph(edges_in)

    # NP-complete, need Grovers algorithm on Quantum Computer
    for k in nx.enumerate_all_cliques(distance_graph):
        complex.append(sum(map(lambda shift: 1 << shift, k)))

    # Since we recently allow num_vertices to be specified, there are vertices that are not part of an edge that need to be added, so here we add all vertices

    for i in range(num_vertices):
        vertex_simplex = 1<<i
        if vertex_simplex not in complex:
            complex.append(vertex_simplex)

    return np.array(complex, dtype=int)
    # return simplices_to_count_vector(np.array(complex, dtype=int), num_vertices)

def vitoris_rips_complex_from_points(points, epsilon):
    """ Returns VR complex as simplices from points"""
    return vitoris_rips_complex_from_one_skeleton(len(points), complex_one_skeleton(points, epsilon))


def vitoris_rips_count_vector_from_pairs_num_pairs_indexed(pairs, num_vertices=None):

    complex = []
    if num_vertices is None:
        num_vertices = int((1+np.sqrt(1+8*len(pairs)))/2)

    return simplices_to_count_vector(vitoris_rips_complex_from_one_skeleton(num_vertices,
        complex_one_skeleton_from_pairs_num_pairs_indexed(pairs)))


def construct_unfilled_triangle():
    # correctly signed
    vit_comp = vitoris_rips_complex_from_points([[0, 0], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)]], 0.51)
    vit_vec = simplices_to_count_vector(vit_comp)

    # print("Vit Complex: ", vit_comp)
    # print("Vit Complex as count vector: ", vit_vec)

    # Adjust individual simplices:
    # print("Removing face")
    vit_vec[5] = -1 # traverse against "the grain"
    vit_vec[7] = 0 # remove filled in face

    return vit_vec

def one_skeleton_triangle():
    return one_skeleton_remap(complex_one_skeleton(
        [[0, 0], [0, 1], [1/np.sqrt(2), 1/np.sqrt(2)]], 0.51))

def construct_unfilled_square():
    # not correctly signed
    vit_comp = vitoris_rips_complex_from_points([[0, 0], [0, 1], [1,0],[1,1]], 0.51)
    vit_vec = simplices_to_count_vector(vit_comp)

    # print("Vit Complex: ", vit_comp)
    # print("Vit Complex as count vector: ", vit_vec)

    return vit_vec

def one_skeleton_square():
    return one_skeleton_remap(complex_one_skeleton([[0, 0], [0, 1], [1,0],[1,1]], 0.51))

def one_skeleton_square_with_diagonal():
    return [[0, 3], [1, 2], [0, 2], [1, 3], [2, 3]], 4 # broke the square picture in mind, but doesn't matter

def construct_unfilled_pyramid():
    #also missing floor traingles
    vit_comp = vitoris_rips_complex_from_points([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 1/np.sqrt(2)]], 0.51)
    vit_vec = simplices_to_count_vector(vit_comp)
    # vit_vec[31] = 0
    # print("Vit Complex: ", vit_comp)
    # print("Vit Complex as count vector: ", vit_vec)

    return vit_vec

def construct_filled_cube():
    #also missing floor traingles
    vit_comp = vitoris_rips_complex_from_points([[0, 0, 0], [0, 1, 0],
                                                 [1, 0, 0], [1, 1, 0],
                                                 [0, 0, 1], [0, 1, 1],
                                                 [1, 0, 1], [1, 1, 1]],
                                                np.sqrt(2)/2+0.1) # eps < sqrt(3)
    
    vit_vec = simplices_to_count_vector(vit_comp)

    return vit_vec


def one_skeleton_unfilled_cube():
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7],
            [0,2],
            [0,5], [1,6], [2,7], [3,4],
            [5,7]], 8


def one_skeleton_two_disconnected_squares(): #b0=2, b1=2, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4]], 8

def one_skeleton_n_disconnected_squares(num_vertices): #b0=n, b1=n, b3=0
    one_square = np.array([[0,1], [1,2], [2,3],[3,0]])
    shift_square = np.array([[[4*s,4*s]] for s in range(num_vertices//4)])
    return (one_square + shift_square).reshape(-1,2).tolist(), num_vertices

def one_skeleton_one_square_four_dangling_vertices(): #b0=1, b1=1, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [0,4], [1,5], [2, 6], [3, 7]], 8


def one_skeleton_two_strongly_connected_squares(): #b0=1, b1=2, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5],
            [0,5], [1,4]], 8


def one_skeleton_three_strongly_connected_squares(): #b0=1, b1=3, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5],
            [0,5], [1,4],
            [2,6]], 8


def one_skeleton_four_weakly_connected_squares(): #b0=1, b1=4, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6]], 8


def one_skeleton_five_strongly_connected_squares(): #b0=1, b1=5, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5],
            [0,5], [1,4],
            [2,6], [3,7]], 8


def one_skeleton_six_weakly_connected_squares(): #b0=1, b1=6, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]], 8


def one_skeleton_two_weakly_connected_squares(): #b0=1, b1=2, b3=0
    return [[0,1], [1,2], [2,3],[3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4]], 8

def construct_unfilled_cube():
    """ Returns VR complex of cube as count vector. """
    return simplices_to_count_vector(vitoris_rips_complex_from_one_skeleton(8,
                                                                            one_skeleton_unfilled_cube()))

def one_skeleton_pyramid():
    return one_skeleton_remap(
        complex_one_skeleton([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0.5, 0.5, 1/np.sqrt(2)]],
                             0.51))

def one_skeleton_tetrahedron():
    return one_skeleton_fully_connected(4)

def construct_unfilled_random():
    points = np.random.rand(6,4)
    vit_comp = vitoris_rips_complex_from_points(points, 0.3)
    vit_vec = simplices_to_count_vector(vit_comp)

    return vit_vec

def one_skeleton_random():
    points = np.random.rand(6, 4)
    return one_skeleton_remap(complex_one_skeleton(points, 0.3))

def one_skeleton_fully_connected(num_vertices):
    return [list(combi) for combi in itertools.combinations(range(num_vertices), 2)], num_vertices

# def classical_boundary_matrix(num_vertices):

#     boundary = boundary_operator(num_vertices)
#     boundary_matrix = classical_matrix_from_pauli_operator(
#         boundary)

#     # print("Boundary matrix:")
#     # print_matrix(boundary_matrix)

#     return boundary_matrix

# def classical_boundary_sum_matrix(num_vertices):

#     boundary_sum = boundary_operator_hermitian(num_vertices)
#     boundary_sum_matrix = matrix_from_pauli_operator(boundary_sum)

#     # print("Boundary sum:")
#     # print_matrix(boundary_sum_matrix, vecfunc=format_basic)

#     return boundary_sum_matrix

# def classical_laplacian_full():
    # print("Boundary sum squared = Laplacian (full):")
    # laplacian_full = boundary_sum_matrix @ boundary_sum_matrix
    # print_matrix(laplacian_full)


def classical_projections_onto_simplices(vit_vec):

    num_vertices = num_underlying_vertices_from_vit_vec(vit_vec)
    # print("Num vertices: ", num_vertices)
    boundary_matrix = classical_boundary_matrix(num_vertices)

    project_on_simplices = np.zeros(
        np.shape(boundary_matrix)+(num_vertices,))

    for i in range(num_vertices):
        project_on_simplices[..., i] = np.array(
            np.diag(d_dim_simplices_mask(i, vit_vec)), dtype=float)
        # print("Project onto simplices in complex of order: ", i)
        # print_matrix(project_on_simplices[..., i])

    return project_on_simplices


def classical_laplacian_of_vit_vec(vit_vec):

    # The internal projections depend on what dimension is being
    # applied to, the boundary sum takes the dimension up and down

    num_vertices = num_underlying_vertices_from_vit_vec(vit_vec)

    project_on_simplices = classical_projections_onto_simplices(
        vit_vec)
    boundary_sum_matrix = classical_boundary_sum_matrix(
        num_vertices)

    laplacian_restricted = np.zeros(np.shape(project_on_simplices))

    for i in range(num_vertices):
        if i > 0:
            project_up_down = project_on_simplices[..., i-1]
        else:
            project_up_down = np.zeros(np.shape(
                project_on_simplices[..., 0]))

        if i < num_vertices-1:
            project_up_down += project_on_simplices[..., i+1]

        laplacian_restricted[..., i] = \
            project_on_simplices[..., i] \
            @ boundary_sum_matrix \
            @ project_up_down \
            @ boundary_sum_matrix \
            @ project_on_simplices[..., i]

        # print("Restricted Laplacian of order: ", i)
        # print_matrix(laplacian_restricted[...,i],
        # vecfunc=format_real(decp=0))

    return laplacian_restricted


def classical_betti_from_exact_kernel_of_boundary(vit_vec):
    # the last kernel and dim_simplices entries [num_vertices]
    # should remain zero, it's needed when calculating betti
    # numbers

    num_vertices = num_underlying_vertices_from_vit_vec(vit_vec)
    boundary_sum_matrix = classical_boundary_sum_matrix(
        num_vertices)

    kernel = np.zeros(num_vertices+1)
    kernel[0] = np.sum(d_dim_simplices_mask(0, vit_vec))

    dim_simplices = np.zeros(num_vertices+1)
    dim_simplices[0] = kernel[0]

    for i in range(num_vertices-1, 0, -1):
        i_dim_simplices = d_dim_simplices_mask(i, vit_vec)
        i_minus_1_dim_simplices = d_dim_simplices_mask(i-1,
                                                       vit_vec)
        dim_simplices[i] = np.sum(i_dim_simplices)
        kernel[i] = dim_simplices[i]

        # Need to check if the complex actually has simplices:
        if np.any(i_minus_1_dim_simplices) and \
           np.any(i_dim_simplices):

            kernel[i] -= np.linalg.matrix_rank(
                boundary_sum_matrix[i_minus_1_dim_simplices]
                [:, i_dim_simplices])

    betti = np.zeros(num_vertices)

    for i in range(num_vertices):
        betti[i] = kernel[i]+kernel[i+1]-dim_simplices[i+1]

    return betti
