// Standalone test for cudnnCTCLoss behavior on impossible sequences.
//
// Compile: nvcc test_cudnn_ctc.cpp -lcudnn -o test_cudnn_ctc
// Run:     ./test_cudnn_ctc

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(expr)                                                    \
  do {                                                                      \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
              cudaGetErrorString(_e), __FILE__, __LINE__);                  \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

#define CHECK_CUDNN(expr)                                                   \
  do {                                                                      \
    cudnnStatus_t _s = (expr);                                              \
    if (_s != CUDNN_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuDNN error: %s at %s:%d\n",                        \
              cudnnGetErrorString(_s), __FILE__, __LINE__);                 \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

// Run cudnnCTCLoss (v7.6 API) for a single batch.
//
// h_probs[T * N * C]: input in log domain; cuDNN applies softmax internally
//   (CUDNN_LOSS_NORMALIZATION_SOFTMAX), so softmax(h_probs) gives the
//   effective per-step label probabilities.  Passing log(uniform) = 0.0
//   gives uniform probabilities 1/C.
// targets: concatenated host label arrays (int32)
// h_costs[N]: filled with per-sample CTC loss on return
// h_grad[T * N * C]: filled with gradient of loss w.r.t. h_probs on return
static void run_cudnn_ctc(
    cudnnHandle_t handle,
    int T, int N, int C,
    const float* h_probs,
    const int* targets,
    const int* input_lengths,
    const int* target_lengths,
    float* h_costs,
    float* h_grad) {

  float *d_probs, *d_grad, *d_costs;
  CHECK_CUDA(cudaMalloc(&d_probs, T * N * C * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_grad,  T * N * C * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_costs, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_probs, h_probs, T * N * C * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudnnCTCLossDescriptor_t ctc_desc;
  CHECK_CUDNN(cudnnCreateCTCLossDescriptor(&ctc_desc));
  // Match PyTorch: SOFTMAX normalization + NaN propagation.
  CHECK_CUDNN(cudnnSetCTCLossDescriptorEx(
      ctc_desc, CUDNN_DATA_FLOAT,
      CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN));

  // PyTorch's TensorDescriptor{t, pad=0} for a 3D tensor uses ndim=3.
  cudnnTensorDescriptor_t probs_desc, grad_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&probs_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&grad_desc));
  int dims[3]    = {T, N, C};
  int strides[3] = {N * C, C, 1};
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(
      probs_desc, CUDNN_DATA_FLOAT, 3, dims, strides));
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(
      grad_desc, CUDNN_DATA_FLOAT, 3, dims, strides));

  size_t workspace_size;
  CHECK_CUDNN(cudnnGetCTCLossWorkspaceSize(
      handle, probs_desc, grad_desc,
      targets, target_lengths, input_lengths,
      CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc,
      &workspace_size));

  void* d_workspace;
  CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size > 0 ? workspace_size : 1));

  CHECK_CUDNN(cudnnCTCLoss(
      handle,
      probs_desc, d_probs,
      targets, target_lengths, input_lengths,
      d_costs,
      grad_desc, d_grad,
      CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, ctc_desc,
      d_workspace, workspace_size));

  CHECK_CUDA(cudaMemcpy(h_costs, d_costs, N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_grad, d_grad, T * N * C * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_probs);
  cudaFree(d_grad);
  cudaFree(d_costs);
  cudaFree(d_workspace);
  cudnnDestroyCTCLossDescriptor(ctc_desc);
  cudnnDestroyTensorDescriptor(probs_desc);
  cudnnDestroyTensorDescriptor(grad_desc);
}

static void print_grad(const float* h_grad, int T, int C) {
  for (int t = 0; t < T; t++) {
    printf("  t=%d:", t);
    for (int c = 0; c < C; c++)
      printf(" %10.6f", h_grad[t * C + c]);
    printf("\n");
  }
}

static bool any_nonzero(const float* data, int n) {
  for (int i = 0; i < n; i++)
    if (data[i] != 0.f) return true;
  return false;
}

int main() {
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));

  // T=2, N=1, C=3.  blank=0 (cuDNN requirement).
  // Uniform inputs: softmax([0,0,...]) = 1/C for each label.
  const int T = 2, N = 1, C = 3;
  std::vector<float> h_probs(T * N * C, 0.f);
  float h_costs[1];
  std::vector<float> h_grad(T * N * C);

  // --- Sanity check: feasible sequence ---
  // target=[1,2], T=2: two distinct labels fit in 2 frames with no blanks.
  printf("=== Feasible: target=[1,2], input_length=2 ===\n");
  {
    int targets[]        = {1, 2};
    int input_lengths[]  = {T};
    int target_lengths[] = {2};
    run_cudnn_ctc(handle, T, N, C, h_probs.data(),
                  targets, input_lengths, target_lengths,
                  h_costs, h_grad.data());
    printf("Loss:           %f  (expected: finite, > 0)\n", h_costs[0]);
    printf("Loss is finite: %s\n", std::isfinite(h_costs[0]) ? "yes" : "no");
    printf("Gradient:\n");
    print_grad(h_grad.data(), T, C);
  }

  // --- Key test: impossible sequence ---
  // target=[1,1], T=2: repeated label requires a blank between them,
  // so the minimum valid alignment needs 3 frames.  No valid CTC path
  // exists, so the mathematical loss is +inf.
  //
  // cuDNN eligibility only checks target_length <= input_length (2 <= 2),
  // so this reaches cudnnCTCLoss.  We want to observe what cuDNN actually
  // returns for both loss and gradient.
  printf("\n=== Impossible: target=[1,1], input_length=2 ===\n");
  printf("    (blank needed between repeated labels; minimum 3 frames)\n");
  {
    int targets[]        = {1, 1};
    int input_lengths[]  = {T};
    int target_lengths[] = {2};
    run_cudnn_ctc(handle, T, N, C, h_probs.data(),
                  targets, input_lengths, target_lengths,
                  h_costs, h_grad.data());
    printf("Loss:             %f\n", h_costs[0]);
    printf("Loss is finite:   %s\n", std::isfinite(h_costs[0]) ? "yes" : "no");
    printf("Loss is zero:     %s\n", h_costs[0] == 0.f ? "yes" : "no");
    printf("Loss is inf:      %s\n", std::isinf(h_costs[0]) ? "yes" : "no");
    printf("Gradient:\n");
    print_grad(h_grad.data(), T, C);
    printf("Gradient non-zero: %s\n",
           any_nonzero(h_grad.data(), T * N * C) ? "yes" : "no");
  }

  cudnnDestroy(handle);
  return 0;
}
