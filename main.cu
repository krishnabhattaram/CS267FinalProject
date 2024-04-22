#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream& fsave, int num_parts, double size) {
    // static bool first = true;

    // if (first) {
    //     fsave << num_parts << " " << size << std::endl;
    //     first = false;
    // }

    // for (int i = 0; i < num_parts; ++i) {
    //     fsave << parts[i].x << " " << parts[i].y << std::endl;
    // }

    //fsave << std::endl;

    return;
}

void load_parameters_toy(char* loadfile, int* n_nodes, int* n_weights,
               int** weights, int** visible_bias, int** hidden_bias) {
    *n_nodes = 3;
    *n_weights = 9;

    *weights = (int*) malloc(*n_weights * sizeof(int));
    *visible_bias = (int*) malloc(*n_nodes * sizeof(int));
    *hidden_bias = (int*) malloc(*n_nodes * sizeof(int));

    // [-9, -12, 4], [-9, 4, -12], [-1, -10, -10]
    // Assumed to be ROW MAJOR
    (*weights)[0] = -9;
    (*weights)[1] = -9;
    (*weights)[2] = -1;

    (*weights)[3] = -12;
    (*weights)[4] = 4;
    (*weights)[5] = -10;

    (*weights)[6] = 4;
    (*weights)[7] = -12;
    (*weights)[8] = -10;

    // [6, 6, 4]
    (*visible_bias)[0] = 6;
    (*visible_bias)[1] = 6;
    (*visible_bias)[2] = 4;

    // [4, 6, 6]
    (*hidden_bias)[0] = 4;
    (*hidden_bias)[1] = 6;
    (*hidden_bias)[2] = 6;

    return;

}

std::vector<std::vector<int>> load_parameters(const char* loadfile, int* n_nodes, int* n_weights,
                     int** weights, int** visible_bias, int** hidden_bias) {
    int temperature = 1;
    int coupling = 10;

    std::ifstream infile(loadfile);

    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open file " << loadfile << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read parameters from the first line
    infile >> *n_nodes >> *n_weights >> *n_weights;
    *n_weights = *n_nodes * *n_nodes;

    *weights = new int[*n_nodes * *n_nodes](); // Initialize with zeros
    *visible_bias = new int[*n_nodes]();
    *hidden_bias = new int[*n_nodes]();

    // Adjacency matrix
    std::vector<std::vector<int>> adj_matrix(*n_nodes, std::vector<int>(*n_nodes, 0));

    int node1, node2;
    float edge;
    while (infile >> node1 >> node2 >> edge) {
        // Convert 1-based indexing to 0-based indexing
        node1--;
        node2--;

        (*weights)[node1 * *n_nodes + node2] = edge;
        (*weights)[node2 * *n_nodes + node1] = edge;

        adj_matrix[node1][node2] = edge;
        adj_matrix[node2][node1] = edge;
    }

    // Set diagonal elements to -1 * coupling
    for (int i = 0; i < *n_nodes; ++i) {
        (*weights)[i * *n_nodes + i] = -1 * coupling;
    }

    // Invert the weight matrix
    for (int i = 0; i < *n_nodes * *n_nodes; ++i) {
        (*weights)[i] *= -1;
    }

    // Calculate visible and hidden biases adjustments
    for (int i = 0; i < *n_nodes; ++i) {
        for (int j = 0; j < *n_nodes; ++j) {
            (*visible_bias)[i] += (*weights)[j * *n_nodes + i];
            (*hidden_bias)[i] += (*weights)[i * *n_nodes + j];
        }
        (*visible_bias)[i] *= temperature * 2;
        (*hidden_bias)[i] *= temperature * 2;
    }

    // Scale weights and biases by temperature
    for (int i = 0; i < *n_nodes * *n_nodes; ++i) {
        (*weights)[i] *= temperature * 4;
    }

    // print weights for debugging
    std::cout << "[";
    for (int i = 0; i < *n_nodes; i++) {
        std::cout << "[";
        for (int j = 0; j < *n_nodes; j++) {
            std::cout << (*weights)[i * *n_nodes + j] << ", ";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]" << std::endl;

    infile.close();

    return adj_matrix;
}

int cut_value(std::vector<std::vector<int>> adj, int num_visible, const int* state) {
    int cut = 0;
    for (int i = 0; i < num_visible; ++i) {
        for (int j = 0; j < i; ++j) {
            if (adj[i][j] != 0 && state[i] != state[j]) {
                cut += 1;
            }
        }
    }
    return cut;
}


// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-r <int>: set number of restarts (how many times to run)" << std::endl;
        std::cout << "-l <int>: set the length of a trial" << std::endl;
        std::cout << "-i <filename>: set the input file with weights and biases" << std::endl;
        std::cout << "-o <filename>: set the output file name" << std::endl;
        std::cout << "-s <int>: set particle initialization seed" << std::endl;
        return 0;
    }

    // Open Output File
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    char* loadname = find_string_option(argc, argv, "-i", nullptr);
    // std::ofstream fsave(savename);

    // Initialize Values
    int n_steps = find_int_arg(argc, argv, "-l", 512);
    int n_trials = find_int_arg(argc, argv, "-r", 1);
    int seed = find_int_arg(argc, argv, "-s", 0);

    int n_nodes;
    int n_weights;

    int* weights;
    int* visible_bias;
    int* hidden_bias;


    std::vector<std::vector<int>> adj_matrix;
    bool clamp = false;
    if (loadname == nullptr) {
        load_parameters_toy(loadname, &n_nodes, &n_weights, &weights, &visible_bias, &hidden_bias);
        clamp = true;
    } else {
        adj_matrix = load_parameters(loadname, &n_nodes, &n_weights, &weights, &visible_bias, &hidden_bias);
    }
    int* visible_vals = (int*) malloc(n_trials * n_steps * n_nodes * sizeof(int));

    int* trial_visible_vals_gpu;
    cudaMalloc((void**)&trial_visible_vals_gpu, n_nodes * n_steps * sizeof(int));

    // Algorithm
    auto start_time = std::chrono::steady_clock::now();
    init_rbm(n_nodes, n_weights, n_steps, weights, visible_bias, hidden_bias, seed);

    for (int trial = 0; trial < n_trials; ++trial) {
        reset_rbm(n_nodes);

        for (int step = 0; step < n_steps; ++step) {
            simulate_one_step(n_nodes, step, seed, trial_visible_vals_gpu, clamp);
        }

        cudaMemcpy(visible_vals + trial * n_steps * n_nodes, trial_visible_vals_gpu, 
                   n_steps * n_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    }

    // Debugging
    for (int i = 0; i < n_steps*n_nodes; i+=n_nodes){
        std::cout << "Step "<< i/n_nodes << " ";
        for (int j = 0; j < n_nodes; j++){
            std::cout << visible_vals[i+j];
        }
        if (loadname != nullptr) {
            std::cout << " Cut Value: " << cut_value(adj_matrix, n_nodes, visible_vals + i);
        }
        std::cout << std::endl;
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds for " << n_nodes << " nodes for " << n_steps << " steps.\n";
    // fsave.close();
    // cudaFree(parts_gpu);
    free(weights);
    free(visible_bias);
    free(hidden_bias);
    free(visible_vals);
    free_rbm();
}

