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
import logging

##########################################################################
# Bit Fiddling
##########################################################################


def first_non_zero_bit(number):
    binary_string_number = integer_to_binary_as_array(number)
    for i, bit in enumerate(binary_string_number):
        if bit == '1':
            return i

def is_power_of_2(number):
    closest_pow_2 = int(np.log2(number))
    return (1 << closest_pow_2 == number)

def num_bits(number):
    """
    Returns the minimum number of bits needed to represent `number` in binary. (no funny business here in terms of usage for counting simplex order, the simplex order convention comes in, in how this function is called.)
    """
    
    number = abs(number)
    if number > 0:
        return int(np.ceil(np.log2(number)))
    else:
        return 1

def integer_to_binary_as_string(number, width=None):
    #right most character is lsb
    if width is None:
        width=num_bits(number)
    return ('{:0'+str(width)+'b}').format(number)
    
def integer_to_binary_as_array(number, width=None):
    #NOTE: RETURNS QISKIT ORDERING, in the sense that first element of array is lsb
    return list(integer_to_binary_as_string(number, width)[::-1])

def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask)

# setBit() returns an integer with the bit at 'offset' set to 1.
def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)

# clearBit() returns an integer with the bit at 'offset' cleared.
def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)

# toggleBit() returns an integer with the bit at 'offset' inverted, 0 -> 1 and 1 -> 0.
def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)

def countBits(int_type, number_bits=32):
    mycount = 0
    for sweep in range(number_bits):
        mycount += (bool)(testBit(int_type, sweep))
    return mycount

#def f(i, d={0:lambda:0, 1:lambda:1, 2:lambda:1, 3:lambda:2}): return d.get(i, lambda: f(i//4) + f(i%4))()

#int popcount64d(uint64_t x)
# {
#     int count;
#     for (count=0; x; count++)
#         x &= x - 1;
#     return count;
# }
# def numberOfSetBits(i):
#     i = i - ((i >> 1) & 0x55555555)
#     i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
#     return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

# import gmpy.popcount
# def bitsoncount(x):
#    return bin(x).count('1')
# https://stackoverflow.com/questions/9829578/fast-way-of-counting-non-zero-bits-in-positive-integer
# numpy.unpackbits?

##########################################################################
# Matrix index Fiddling
##########################################################################
 #https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist

import math

def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 * n**2 - 4*n - 7)**0.5 + 2*n - 1) - 1))


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))/2


def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j

# Haha, I rederived the above, I think. TODO: Consolidate

def pair_index_to_two_power_2s(k): #0-index
    p1 = int(np.floor((np.sqrt(8*(k+1) - 7)-1)/2) + 1)
    p2 = int(k - (p1-1)*(p1)/2)
    # Use the following to create test:
    # print(p1, p2)
    # pair_str = list(integer_to_binary((1<<p1) + (1<<p2)))
    # pair_str.reverse()
    # print("{:02}".format(k) + ': ' + "0"*(num_qubits-p1-1)+"".join(pair_str))
    return p1, p2

def two_power_2s_to_pair_index(p1, p2):
    # p1 = n-1
    # p2 = n-2

    # (n-2)*(n-1)/2 + n-2
    # = (n-2)((n+1)/2)

    # n*(n-1)/2 - 1
    # = (n^2 -n -2)/2
    # (n-2)(n+1)/2
    # Use the following to create test:
    # self.assertEqual(k, two_power_2s_to_pair_index(p1, p2))
    return int((p1-1)*(p1)/2 + p2)


def num_qubits_from_num_pairs(pairs):
    # pairs to qubits
    num_qubits = int((1+np.sqrt(8*pairs+1))/2)
    if pairs != int(num_qubits*(num_qubits-1)/2):
        raise ValueError("Number of pairs incorrect (!= n(n-1)/2).")

    return num_qubits


##########################################################################
# Matrix Printing Fiddling
##########################################################################

import numpy as np
#import pandas as pd

def print_sign_single(x):
    if np.isclose(x,0):
        mychar = '.'
    elif x < 0:
        mychar="-"
    else:
        mychar="+"
    return mychar

print_signs = np.vectorize(print_sign_single)

def binarize_single(x):
    if x < 0:
        mychar=0
    else:
        mychar=1
    return mychar

binarize = np.vectorize(binarize_single)

def format_basic_single(x, decp=1):
    return ('{: .'+str(decp)+'f}').format(np.round(x,decp))

format_basic_v0 = np.vectorize(format_basic_single, excluded=['decp'])

def format_basic(x=None, decp=1):
    if x is None:
        return lambda v: format_basic_v0(v, decp=decp)
    else:
        return format_basic_v0(x, decp=decp)

def format_complex_single(x, decp=1):
    return ('{0.real: .'+str(decp)+'f}{0.imag:+.'+str(decp)+'f}i').format(x)

format_complex_v0 = np.vectorize(format_complex_single)

def format_complex(x=None, decp=1):
    if x is None:
        return lambda v: format_complex_v0(v, decp=decp)
    else:
        return format_complex_v0(x, decp=decp)

def format_real_single(x, decp=1):
    return ('{0.real: .'+str(decp)+'f}').format(x)

format_real_v0 = np.vectorize(format_real_single)

def format_real(x=None, decp=1):
    if x is None:
        return lambda v: format_real_v0(v, decp=decp)
    else:
        return format_real_v0(x, decp=decp)

def format_imag_single(x, decp=1):
    return ('{0.imag: .'+str(decp)+'f}').format(x)

format_imag_v0 = np.vectorize(format_imag_single)

def format_imag(x=None, decp=1):
    if x is None:
        return lambda v: format_imag_v0(v, decp=decp)
    else:
        return format_imag_v0(x, decp=decp)

def print_matrix(matrix, vecfunc=format_basic):
    """Print matrices, running each element through vecfunc (default identity)."""
    print(*map(lambda x: " ".join(x), vecfunc(matrix).tolist()), sep='\n')

def log_matrix(matrix, vecfunc=format_basic):
    """Print matrices, running each element through vecfunc (default identity)."""
    logging.info('\n'.join(list(map(lambda x: " ".join(x), vecfunc(matrix).tolist()))))

def print_matrix_and_info(matrix, vecfunc=format_basic):
    eigvals, eigenvecs = np.linalg.eig(matrix)
    print("Matrix: ")
    print_matrix(matrix, vecfunc=vecfunc)
    print("Rank: ", np.linalg.matrix_rank(matrix))
    print("Eigenvalues: ")
    print_matrix(eigvals.reshape((1,-1)), vecfunc=vecfunc)
    print("Eigenvectors: ")
    print_matrix(eigenvecs, vecfunc=vecfunc)

    #https://stackoverflow.com/questions/41069825/convert-binary-01-numpy-to-integer-or-binary-string

#<64 bits
# def bin2int(bits):
#     return np.right_shift(
#         np.packbits(bits, -1), bits.size).squeeze()

def bin_string2int(bitstring):
    """Convert a binary boolean string (not qiskit ordering, msb is first) into integer."""

    if isinstance(bitstring, list):
        return list(
            map(lambda innerstring: bin_array2int(list(map(int, list(innerstring[::-1])))), bitstring))
    else:
        return bin_array2int(list(map(int, list(bitstring[::-1]))))

def bin_array2int(bits):
    """Convert a binary boolean array (qiskit ordering, lsb is first) into integer."""
    total = 0
    for shift, j in enumerate(bits):
        if j:
            total += 1 << shift
    return total

def bin_index2int(indices):
    """Convert array of indices of a binary array that contain 1."""
    return sum(map(lambda x:1<<x, indices))
