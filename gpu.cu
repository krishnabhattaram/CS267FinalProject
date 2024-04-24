#include "common.h"
#include <curand_kernel.h>
#include <curand.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>

#define NUM_THREADS 1024

float* weights_gpu;
float* visible_bias_gpu;
float* hidden_bias_gpu;

int* intermediate_visible_states;
int* visible_state_gpu;
int* hidden_state_gpu;
curandState_t* states;

__global__ void visible_to_hidden_gpu(int* visible_state, int* hidden_state, float* weights, float* visible_bias,
                                      float* hidden_bias, int step, int n_nodes, int n_blocks, curandState_t* states) {
    __shared__ int dot_arr[NUM_THREADS];

    int num_nodes_per_block = (n_nodes + n_blocks - 1) / n_blocks;
    for (int i = 0; i < num_nodes_per_block; i++) {
        int node_idx = i + num_nodes_per_block * blockIdx.x;
        if (node_idx >= n_nodes) {
            return;
        }

        // Initialize shared memory
        dot_arr[threadIdx.x] = 0;

        for (int j = 0; j < (n_nodes + NUM_THREADS - 1) / NUM_THREADS; j++) {
            int idx = j * NUM_THREADS + threadIdx.x;
            if (idx < n_nodes) {
                dot_arr[threadIdx.x] += weights[node_idx * n_nodes + idx] * visible_state[idx];
            }
        }

        // __syncthreads();

        // Reduction to compute final dot product
        for (unsigned int s = 1; s < NUM_THREADS; s *= 2) {
            int index = 2 * s * threadIdx.x;
            if (index < NUM_THREADS) {
                dot_arr[index] += dot_arr[index + s];
            }
            // __syncthreads();
        }

        if (threadIdx.x == 0) {
            int dot_product = dot_arr[0] + hidden_bias[node_idx];
            // printf("BlockIdx %d Visible %d%d%d\n", blockIdx.x, visible_state[0], visible_state[1], visible_state[2]);
            // printf("BlockIdx %d V2H Dot Product %d\n\n", blockIdx.x, dot_product);

            double sigmoid = 1. / (1 + exp(-1.0 * dot_product));

            float randunif = curand_uniform(&states[node_idx]);
            hidden_state[node_idx] = (sigmoid > randunif) ? 1 : 0;
        }
    }

    return;
}

__global__ void hidden_to_visible_gpu(int* visible_state, int* hidden_state, float* weights, float* visible_bias,
                                      float* hidden_bias, int step, int n_nodes, int n_blocks, curandState_t* states, bool clamp) {
    __shared__ int dot_arr[NUM_THREADS];

    int num_nodes_per_block = (n_nodes + n_blocks - 1) / n_blocks;
    for (int i = 0; i < num_nodes_per_block; i++) {
        int node_idx = i + num_nodes_per_block * blockIdx.x;
        if (node_idx >= n_nodes) {
            return;
        }

        for (int j = 0; j < (n_nodes + NUM_THREADS - 1) / NUM_THREADS; j++) {
            int idx = j * NUM_THREADS + threadIdx.x;
            if (idx < n_nodes) {
                dot_arr[threadIdx.x] += weights[node_idx + n_nodes * idx] * visible_state[idx];
            }
        }

        // __syncthreads();

        // Reduction to compute final dot product
        for (unsigned int s = 1; s < NUM_THREADS; s *= 2) {
            int index = 2 * s * threadIdx.x;
            if (index < NUM_THREADS) {
                dot_arr[index] += dot_arr[index + s];
            }
            // __syncthreads();
        }

        if (threadIdx.x == 0) {
            int dot_product = dot_arr[0] + visible_bias[node_idx];
            // printf("BlockIdx %d Hidden %d%d%d\n", blockIdx.x, hidden_state[0], hidden_state[1], hidden_state[2]);
            // printf("BlockIdx %d H2V Final Dot Product %d\n\n", blockIdx.x, dot_product);

            double sigmoid = 1. / (1 + exp(-1.0 * dot_product));

            float randunif = curand_uniform(&states[node_idx]);
            visible_state[node_idx] = (sigmoid > randunif) ? 1 : 0;

            if (clamp) {
                // Clamping
                if (node_idx == 2) {
                    visible_state[node_idx] = 1;
                }
            }
        }
    }

    return;
}

__global__ void init_states(int seed, curandState_t* states) {
    // Initialize the state using the provided seed and block id as the sequence number
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void randomize_visible_state(int* visible_state_gpu, int n_nodes, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_nodes) {
        float rand_num = curand_uniform(&state[idx]);
        visible_state_gpu[idx] = (rand_num < 0.5) ? 0 : 1;
    }
}

void init_rbm(int n_nodes, int n_weights, int n_steps, float* weights, float* visible_bias, float* hidden_bias, int seed) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    cudaMalloc((void**)&weights_gpu, n_weights * sizeof(float));
    cudaMemcpy(weights_gpu, weights, n_weights * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&visible_bias_gpu, n_nodes * sizeof(float));
    cudaMemcpy(visible_bias_gpu, visible_bias, n_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&hidden_bias_gpu, n_nodes * sizeof(float));
    cudaMemcpy(hidden_bias_gpu, hidden_bias, n_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&intermediate_visible_states, n_steps * sizeof(int));
    cudaMalloc((void**)&visible_state_gpu, n_nodes * sizeof(int));
    cudaMalloc((void**)&hidden_state_gpu, n_nodes * sizeof(int));

    cudaMalloc((void**)&states, n_nodes * sizeof(curandState_t));
    init_states<<<n_nodes, NUM_THREADS>>>(seed, states);
}

void reset_rbm(int n_nodes) {
    // Resetting according to Jupyter notebook to all 1s
    // cudaMemset(visible_state_gpu, 0, n_nodes*sizeof(int));
    // Initialize random number generator states
    // Launch kernel to randomize visible_state_gpu
    randomize_visible_state<<<(n_nodes + 255) / 256, 256>>>(visible_state_gpu, n_nodes, states);
    cudaDeviceSynchronize();
    // print visible_state_gpu
    int* visible_state = (int*) malloc(n_nodes * sizeof(int));
    cudaMemcpy(visible_state, visible_state_gpu, n_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n_nodes; i++) {
    //     std::cout << visible_state[i];
    // }
    // std::cout << std::endl;
}

void free_rbm() {
    // Free any data objects you may have allocated
    cudaFree(weights_gpu);
    cudaFree(visible_bias_gpu);
    cudaFree(hidden_bias_gpu);
    cudaFree(intermediate_visible_states);
    cudaFree(visible_state_gpu);
    cudaFree(hidden_state_gpu);
    cudaFree(states);
}

void simulate_one_step(int n_blocks, int n_nodes, int step, int seed, int* trial_visible_vals_gpu, bool clamp) {
    visible_to_hidden_gpu<<<n_blocks, NUM_THREADS>>>(visible_state_gpu, hidden_state_gpu, weights_gpu, 
                                                    visible_bias_gpu, hidden_bias_gpu, step, n_nodes, n_blocks, states);
    cudaDeviceSynchronize();
    hidden_to_visible_gpu<<<n_blocks, NUM_THREADS>>>(visible_state_gpu, hidden_state_gpu, weights_gpu, 
                                                    visible_bias_gpu, hidden_bias_gpu, step, n_nodes, n_blocks, states, clamp);
    cudaDeviceSynchronize();
    
    // Log current visible node values
    cudaMemcpy(trial_visible_vals_gpu + step * n_nodes, visible_state_gpu, n_nodes * sizeof(int), cudaMemcpyDeviceToDevice);
}
