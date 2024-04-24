#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Simulation routine
void init_rbm(int n_nodes, int n_weights, int n_steps, float* weights, float* visible_bias, float* hidden_bias, int seed);
void reset_rbm(int n_nodes);
void simulate_one_step(int n_blocks, int n_nodes, int step, int seed, int* trial_visible_vals_gpu, bool clamp);
void free_rbm();
#endif
