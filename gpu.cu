#include "common.h"
#include <curand_kernel.h>
#include <curand.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>

#define NUM_THREADS 1024

int* weights_gpu;
int* visible_bias_gpu;
int* hidden_bias_gpu;

int* intermediate_visible_states;
int* visible_state_gpu;
int* hidden_state_gpu;

__global__ void visible_to_hidden_gpu(int* visible_state, int* hidden_state, int* weights, int* visible_bias,
                                      int* hidden_bias, int step, int n_nodes, int seed) {
    int node_idx = blockIdx.x;
    __shared__ int dot_arr[NUM_THREADS];
    dot_arr[threadIdx.x] = 0;

    // NUM_THREADS must be greater than n_nodes!
    if (threadIdx.x < n_nodes) {
        dot_arr[threadIdx.x] = weights[node_idx * n_nodes + threadIdx.x] * visible_state[threadIdx.x];
    }

    __syncthreads();

    // NUM_THREADS must be even!
    // Reduce element products to final dot product
    for (unsigned int s = 1; s < NUM_THREADS; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            dot_arr[threadIdx.x] += dot_arr[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int dot_product = dot_arr[0] + hidden_bias[node_idx];
        // printf("BlockIdx %d Visible %d%d%d\n", blockIdx.x, visible_state[0], visible_state[1], visible_state[2]);
        // printf("BlockIdx %d V2H Dot Product %d\n\n", blockIdx.x, dot_product);

        double sigmoid = 1. / (1 + exp(-1.0 * dot_product));

        curandState state;
        curand_init(seed, step, 0, &state);
        float randunif = curand_uniform(&state);
        hidden_state[node_idx] = (sigmoid > randunif) ? 1 : 0;
    }

    return;
}

__global__ void hidden_to_visible_gpu(int* visible_state, int* hidden_state, int* weights, int* visible_bias,
                                      int* hidden_bias, int step, int n_nodes, int seed) {
    int node_idx = blockIdx.x;
    __shared__ int dot_arr[NUM_THREADS];
    dot_arr[threadIdx.x] = 0;

    // NUM_THREADS must be greater than n_nodes!
    if (threadIdx.x < n_nodes) {
        dot_arr[threadIdx.x] = hidden_state[threadIdx.x] * weights[threadIdx.x * n_nodes + node_idx];
    }

    __syncthreads();

    // NUM_THREADS must be even!
    // Reduce element products to final dot product
    for (unsigned int s = 1; s < NUM_THREADS; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            dot_arr[threadIdx.x] += dot_arr[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int dot_product = dot_arr[0] + visible_bias[node_idx];
        // printf("BlockIdx %d Hidden %d%d%d\n", blockIdx.x, hidden_state[0], hidden_state[1], hidden_state[2]);
        // printf("BlockIdx %d H2V Final Dot Product %d\n\n", blockIdx.x, dot_product);

        double sigmoid = 1. / (1 + exp(-1.0 * dot_product));

        curandState state;
        curand_init(seed, step, 0, &state);
        float randunif = curand_uniform(&state);
        visible_state[node_idx] = (sigmoid > randunif) ? 1 : 0;

        // Clamping
        if (node_idx == 2) {
            visible_state[node_idx] = 1;
        }
    }

    return;
}

void init_rbm(int n_nodes, int n_weights, int n_steps, int* weights, int* visible_bias, int* hidden_bias) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    cudaMalloc((void**)&weights_gpu, n_weights * sizeof(int));
    cudaMemcpy(weights_gpu, weights, n_weights * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&visible_bias_gpu, n_nodes * sizeof(int));
    cudaMemcpy(visible_bias_gpu, visible_bias, n_nodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&hidden_bias_gpu, n_nodes * sizeof(int));
    cudaMemcpy(hidden_bias_gpu, hidden_bias, n_nodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&intermediate_visible_states, n_steps * sizeof(int));
    cudaMalloc((void**)&visible_state_gpu, n_nodes * sizeof(int));
    cudaMalloc((void**)&hidden_state_gpu, n_nodes * sizeof(int));
}

void reset_rbm(int n_nodes) {
    // Resetting according to Jupyter notebook to all 1s (should be randomized for max cut)
    cudaMemset(visible_state_gpu, 0, n_nodes*sizeof(int));
}


void simulate_one_step(int n_nodes, int step, int seed, int* trial_visible_vals_gpu) {
    visible_to_hidden_gpu<<<n_nodes, NUM_THREADS>>>(visible_state_gpu, hidden_state_gpu, weights_gpu, 
                                                    visible_bias_gpu, hidden_bias_gpu, step, n_nodes, seed);
    cudaDeviceSynchronize();
    hidden_to_visible_gpu<<<n_nodes, NUM_THREADS>>>(visible_state_gpu, hidden_state_gpu, weights_gpu, 
                                                    visible_bias_gpu, hidden_bias_gpu, step, n_nodes, seed);
    cudaDeviceSynchronize();
    
    // Log current visible node values
    cudaMemcpy(trial_visible_vals_gpu + step * n_nodes, visible_state_gpu, n_nodes * sizeof(int), cudaMemcpyDeviceToDevice);
}
