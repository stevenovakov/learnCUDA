/*
# summer.cu
#     for learnCUDA
#     Copyright (C) 2015 Steve Novakov

#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "summer.h"

// Macro to catch CUDA errors in kernel launches
// from user njuffa on nvidia developer forums
// https://devtalk.nvidia.com/default/topic/865548/
// cuda-programming-and-performance/problem-lanuching-simple-kernel/
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error (sunc) in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error (async) in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// summing kernel
__global__ void Summer(
  float * one,
  float * two,
  float * out
)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  out[index] = one[index] + two[index];
}

// c++ wrapper
void runSummer(
  dim3 grid,
  dim3 blocks,
  float * input_one_d,
  float * intput_two_d,
  float * output_d,
  cudaStream_t * stream
)
{
  Summer<<<grid, blocks, 0, *stream>>>(input_one_d, intput_two_d, output_d);
  //CHECK_LAUNCH_ERROR();
}
