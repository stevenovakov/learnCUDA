/*
# main.cc
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
#include <string>
#include <random>

#include <stdio.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "summer.h"
#include "customtypes.h"

ConfigData config = {
  100.0,  // output data size (MB)
  10.0,   // processing chunk size per enqueueNDRangeKernel call (MB)
  std::vector<uint32_t>() // specific gpus to use, if empty: use all available.
};

void CLArgs(int argc, char * argv[]);

int main(int argc, char * argv[])
{
  int gpus;
  cudaGetDeviceCount(&gpus);

  for (int d = 0; d < gpus; d++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, d);

    printf("Device Number: %d\n", d);
    printf("Device name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("asyncEngineCount: %d\n", prop.asyncEngineCount);
    printf("Total Global Memory (kB): %lu\n",
      prop.totalGlobalMem/1000);
    printf("Max Device Memory Pitch (kB): %lu\n",
      prop.memPitch/1000);
    printf("Max Grid Size (%d, %d, %d)\n",
      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Block Size (%d, %d, %d)\n\n",
      prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  }

  // Hanndle CLI parameters, if any

  CLArgs(argc, argv);

  // Set up I/O containers and fill input.

  float fraction = config.data_size / config.chunk_size;
  float rem = fraction - static_cast<uint32_t>(fraction);

  if ( rem > 0.0 || fraction < 1.0)
  {
    puts("GPU data size must be an integer multiple of the chunk size, \
      as padding is unsupported.");
    return 0;
  }

  float total_size = config.data_size;
  // total size of each input array, in MB
  float chunk_size = config.chunk_size;
  // size of each chunk summed by a single kernel execution

  uint32_t n = static_cast<uint32_t>(total_size * 1e6 / sizeof(float));
  uint32_t n_gpu = static_cast<uint32_t>(config.data_size * 1e6 /
    sizeof(float));
  uint32_t n_chunk = static_cast<uint32_t>( chunk_size * 1e6 / sizeof(float));
  uint32_t n_chunks = n_gpu / n_chunk;

  printf("Total Input Size: %.3f (MB), GPU Size: %.3f (MB), \
    Compute Chunk: %.3f (MB), Total Array Size: %d, GPU Array Size: %d\n",
      total_size, config.data_size, chunk_size, n, n_gpu);

  std::vector<float> input_one, input_two, output;

  input_one.resize(n);
  input_two.resize(n);
  output.resize(n);

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  puts("Generating random number sets...\n");
  for (uint32_t i = 0; i < n; i++)
  {
    input_one.at(i) = distribution(generator);
    input_two.at(i) = distribution(generator);
  }
  puts("Number sets complete.\n");

  uint32_t buffer_mem_size = n_chunk * sizeof(float);

  printf("N Chunks: %d, Chunk Buffer Size: %d (B)\n",
    n_chunks, buffer_mem_size);

  // CUDA memory transfers and kernel execution

  float * input_one_d, * input_two_d, * output_d;

  cudaError_t err;
  cudaSetDevice(0);

  err = cudaMalloc((void **) &input_one_d, buffer_mem_size);
  if (err != cudaSuccess)
      printf("Error (malloc 1): %s\n", cudaGetErrorString(err));
  err = cudaMalloc((void **) &input_two_d, buffer_mem_size);
  if (err != cudaSuccess)
    printf("Error (malloc 2): %s\n", cudaGetErrorString(err));
  err = cudaMalloc((void **) &output_d, buffer_mem_size);
  if (err != cudaSuccess)
      printf("Error (malloc 3): %s\n", cudaGetErrorString(err));

  dim3 grid(n_chunk);
  dim3 blocks(1);

  for (uint32_t c = 0; c < n_chunks; c++)
  {
    err = cudaMemcpy((void*) input_one_d, input_one.data() + (c * n_chunk),
      buffer_mem_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      printf("Error (memcpy htod 1): %s\n", cudaGetErrorString(err));
    err = cudaMemcpy((void*) input_two_d, input_two.data() + (c * n_chunk),
      buffer_mem_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      printf("Error (memcpy htod 2): %s\n", cudaGetErrorString(err));

    runSummer(grid, blocks, input_one_d, input_two_d, output_d);

    err = cudaMemcpy(output.data() + (c * n_chunk), (void*) output_d,
      buffer_mem_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      printf("Error (memcpy dtoh 1): %s\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();
  printf("100.00%% complete\n");

  // random tests of correctness

  uint32_t n_tests = 20;

  printf("Testing %d random entries for correctness...\n", n_tests);

  std::uniform_int_distribution<uint32_t> int_distro(0, n);

  for (uint32_t i = 0; i < n_tests; i++)
  {
    uint32_t entry = int_distro(generator);

    printf("Entry %d -> %.4f + %.4f = %.4f ? %.4f\n", entry,
      input_one.at(entry), input_two.at(entry), output.at(entry),
        input_one.at(entry) + input_two.at(entry));
  }

  // cleanup

  cudaFree(input_one_d);
  cudaFree(input_two_d);
  cudaFree(output_d);

  return 0;
}

void CLArgs(int argc, char * argv[])
{
  std::vector<std::string> args(argv, argv+argc);

  for (uint32_t i = 0; i < args.size(); i++)
  {
    if (args.at(i).find("-datasize") == 0)
    {
      config.data_size=std::stof(args.at(i).substr(args.at(i).find('=')+1));
    }
    else if (args.at(i).find("-chunksize") == 0)
    {
      config.chunk_size=std::stof(args.at(i).substr(args.at(i).find('=')+1));
    }
    else if (args.at(i).find("-gpus") == 0)
    {
      std::string delim = ",";

    }
  }
}

//EOF
