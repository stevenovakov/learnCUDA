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
  bool overlap = true;
  std::vector<cudaDeviceProp> props(gpus);

  for (int d = 0; d < gpus; d++) {
    cudaGetDeviceProperties(&props.at(d), d);

    printf("Device Number: %d\n", d);
    printf("Device name: %s\n", props.at(d).name);
    printf("Compute Capability: %d.%d\n", props.at(d).major,
      props.at(d).minor);
    printf("asyncEngineCount: %d\n", props.at(d).asyncEngineCount);
    printf("Total Global Memory (kB): %lu\n",
      props.at(d).totalGlobalMem/1000);
    printf("Max Device Memory Pitch (kB): %lu\n",
      props.at(d).memPitch/1000);
    printf("Max Grid Size (%d, %d, %d)\n",
      props.at(d).maxGridSize[0], props.at(d).maxGridSize[1],
        props.at(d).maxGridSize[2]);
    printf("Max Block Size (%d, %d, %d)\n\n",
      props.at(d).maxThreadsDim[0], props.at(d).maxThreadsDim[1],
        props.at(d).maxThreadsDim[2]);
    printf("Warp Size: %d\n", props.at(d).warpSize);
    printf("Max Threads per SMP: %d\n\n",
      props.at(d).maxThreadsPerMultiProcessor);

    overlap &= props.at(d).deviceOverlap;
  }

  if (!overlap)
  {
    puts("prop.deviceOverlap not TRUE for all devices. Exiting...");
    return 0;
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

  float total_size = config.data_size * gpus;
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

  float * input_one, * input_two, * output;

  cudaMallocHost((void **) &input_one, n * sizeof(float));
  cudaMallocHost((void **) &input_two, n * sizeof(float));
  cudaMallocHost((void **) &output, n * sizeof(float));

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  puts("Generating random number sets...\n");

  for (uint32_t i = 0; i < n; i++)
  {
    input_one[i] = distribution(generator);
    input_two[i] = distribution(generator);
  }

  puts("Number sets complete.\n");

  // Calculate pad for buffers so that block size can be multiple of 1024
  // (which is a multiple of 32, the warp size)

  std::vector<uint32_t> pads(gpus);
  std::vector<uint32_t> buffer_allocate_sizes(gpus);
  uint32_t buffer_io_size = n_chunk * sizeof(float);

  std::vector<dim3> grids;
  std::vector<dim3> blocks;

  for (int d = 0; d < gpus; d++)
  {
    pads.at(d) = n_chunk % props.at(d).maxThreadsPerBlock;

    grids.push_back(
      dim3((n_chunk + pads.at(d)) / props.at(d).maxThreadsPerBlock));
    blocks.push_back(dim3(props.at(d).maxThreadsPerBlock));

    buffer_allocate_sizes.at(d) = (n_chunk + pads.at(d)) * sizeof(float);

    printf("N Chunks: %d, Chunk Buffer Size (Device): %d (B) (%d) \n",
      n_chunks, buffer_allocate_sizes.at(d), d);
  }

  //
  // Allocate device buffer memory
  //

  std::vector<cudaStream_t> streams(gpus);

  std::vector<float*> ones(gpus);
  std::vector<float*> twos(gpus);
  std::vector<float*> outs(gpus);

  cudaError_t err;

  for (int d = 0; d < gpus; d++)
  {
    cudaSetDevice(d);

    cudaStreamCreate(&(streams.at(d)));

    err = cudaMalloc((void **) &(ones.at(d)), buffer_allocate_sizes.at(d));
    if (err != cudaSuccess)
        printf("Error (malloc 1): %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &(twos.at(d)), buffer_allocate_sizes.at(d));
    if (err != cudaSuccess)
      printf("Error (malloc 2): %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **) &(outs.at(d)), buffer_allocate_sizes.at(d));
    if (err != cudaSuccess)
        printf("Error (malloc 3): %s\n", cudaGetErrorString(err));
  }

  //
  // MemCpy and Kernel execution
  //

  for (uint32_t c = 0; c < n_chunks; c++)
  {
    for (int d = 0; d < gpus; d++)
    {
      cudaSetDevice(d);
      err = cudaMemcpyAsync((void*) ones.at(d),
        input_one + (d * n_gpu) + (c * n_chunk),
          buffer_io_size, cudaMemcpyHostToDevice, streams.at(d));
      if (err != cudaSuccess)
        printf("Error (memcpy htod 1): %s\n", cudaGetErrorString(err));
      err = cudaMemcpyAsync((void*) twos.at(d),
        input_two + (d * n_gpu) + (c * n_chunk),
          buffer_io_size, cudaMemcpyHostToDevice, streams.at(d));
      if (err != cudaSuccess)
        printf("Error (memcpy htod 2): %s\n", cudaGetErrorString(err));
    }

    for (int d = 0; d < gpus; d++)
    {
      cudaSetDevice(d);

      runSummer(grids.at(d), blocks.at(d),
        ones.at(d), twos.at(d), outs.at(d), &(streams.at(d)));
    }

    for (int d = 0; d < gpus; d++)
    {
      cudaSetDevice(d);
      err = cudaMemcpyAsync(output + (d * n_gpu) + (c * n_chunk),
        (void*) outs.at(d), buffer_io_size, cudaMemcpyDeviceToHost,
          streams.at(d));
      if (err != cudaSuccess)
        printf("Error (memcpy dtoh 1): %s\n", cudaGetErrorString(err));
    }
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
      input_one[entry], input_two[entry], output[entry],
        input_one[entry] + input_two[entry]);
  }

  // cleanup
  for (int d = 0; d < gpus; d++)
  {
    cudaSetDevice(d);
    cudaFree(ones.at(d));
    cudaFree(twos.at(d));
    cudaFree(outs.at(d));

    cudaStreamDestroy(streams.at(d));

    cudaDeviceReset();
  }

  cudaFree(input_one);
  cudaFree(input_two);
  cudaFree(output);

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
