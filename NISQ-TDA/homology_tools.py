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
from math_fiddling import countBits, print_matrix, print_signs, num_bits


def num_underlying_vertices_simplices(simplices):
    return num_bits(np.amax(simplices)+1) #added the +1 because num_bits "encodes" from 0


def num_underlying_vertices_vector_length(lenvec):
    num_vertices = int(np.log2(lenvec))
    if (lenvec != 1 << num_vertices):
        raise Exception(
            "Error: count_vector should be a multiple of 2")
    return num_vertices


def num_underlying_vertices_from_vit_vec(vit_vec):
    num_basis = len(vit_vec)
    num_vertices = num_underlying_vertices_vector_length(num_basis)
    return num_vertices


def simplices_to_count_vector(simplices, num_vertices=None):

    if len(simplices) == 0:
        if num_vertices is None:
            raise Exception(
            "Need at least non-empty simplices or num_vertices.")
        else:
            return np.zeros((1 << num_vertices))

    absolute_simplices = np.abs(simplices, dtype=int)
    sign_simplices = np.sign(simplices, dtype=int)

    if num_vertices is None:
        num_vertices = num_underlying_vertices_simplices(absolute_simplices)

    count_vector_simplices = np.zeros((1 << num_vertices))

    for simplex_index, simplex_sign in zip(absolute_simplices,
                                            sign_simplices):
        count_vector_simplices[simplex_index] += simplex_sign

    return count_vector_simplices

def count_vector_to_simplices(count_vector_simplices):
    """ Returns the number of counts as well as the simplices. """
    simplices = np.nonzero(count_vector_simplices)[0]
    return count_vector_simplices[simplices], simplices

def d_dim_simplices_mask(d, count_vector):
# points are zero dimensional
# lines are one dimensional
# perhaps check if the vector element is zero supposedly saves us from calling countBits everywhere
    d += 1
    max_simplices = len(count_vector)
    bit_mask = np.zeros(max_simplices, dtype=bool)

    for i in range(max_simplices):
        if (count_vector[i] != 0) & (countBits(i) == d):
            bit_mask[i] = 1

    return bit_mask # bit_mask * count_vector to apply bit_mask

def d_dim_simplices(d, count_vector):
    return count_vector_to_simplices(d_dim_simplices_mask(d, count_vector))[1]

def reshape_vector_to_matrix_and_index(vec, new_column_size, indices=(Ellipsis)):
    return vec.reshape((new_column_size,-1),order='F')[indices]

def print_signs_simplicial_vector_parts(vec, new_column_size, simplices_dim=None, offset=0):
    """When simplices_dim is not None, we use d_dim_simplices_mask to fetch all the indices that correspond to simplices_dim simplices and pass those through print_signs. We fold the long vec into columns of new_column_size (often double 2^num_vertices for Hermitian), then we allow an offset to focus on second half. Should possibly allow a from and a to offset?"""
    if simplices_dim is None:
        indices = (Ellipsis)
    else:
        indices = (d_dim_simplices_mask(simplices_dim, np.ones(new_column_size-offset)).nonzero()[0]+offset).tolist()

    print_matrix(reshape_vector_to_matrix_and_index(vec, new_column_size, indices), vecfunc=print_signs)

if __name__ == '__main__':
    # from homology_tools import print_signs_simplicial_vector_parts
    from classical_homology import construct_unfilled_triangle
    print()
    print_signs_simplicial_vector_parts(construct_unfilled_triangle(),8,simplices_dim=1)
