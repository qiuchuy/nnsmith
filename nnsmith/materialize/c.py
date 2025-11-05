from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsOpBase
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.torch import TorchModel


class CModel(Model):
    """C model that emits C code for neural network operators."""

    def __init__(self, ir: GraphIR):
        super().__init__()
        self.ir = ir
        # Use TorchModelCPU for oracle generation
        from nnsmith.materialize.torch import TorchModelCPU
        self.torch_model = TorchModelCPU.from_gir(ir)
        self._input_like = self.torch_model.input_like
        self._output_like = self.torch_model.output_like

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return self._input_like

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return self._output_like

    @classmethod
    def from_gir(cls, ir: GraphIR, **kwargs) -> "CModel":
        return cls(ir)

    @classmethod
    def load(cls, path) -> "CModel":
        with open(path, "rb") as f:
            ir = pickle.load(f)
        return cls(ir)

    def dump(self, path) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.ir, f)

    @property
    def native_model(self) -> GraphIR:
        return self.ir

    @staticmethod
    def name_suffix() -> str:
        return ".cmodel"

    def refine_weights(self) -> None:
        # C model doesn't need weight refinement
        pass

    def make_oracle(self) -> Oracle:
        return self.torch_model.make_oracle()

    @staticmethod
    def operators() -> List[type[AbsOpBase]]:
        return TorchModel.operators()

    @property
    def import_libs(self) -> List[str]:
        return [
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "#include <math.h>",
            "#include <time.h>",
            "#include <stdbool.h>",
        ]

    def emit_def(self, mod_name: str, mod_cls: str) -> str:
        """Emit C function definitions for all operators in the graph."""
        return self._emit_operator_functions(mod_name)

    def emit_run(self, out_name: str, mod_name: str, inp_name: str) -> str:
        """Emit C code to run the computational graph."""
        return self._emit_graph_execution(out_name, mod_name, inp_name)

    def emit_weight(self, mod_name: str, path: Optional[os.PathLike] = None) -> str:
        """Emit C code to initialize model weights."""
        return self._emit_weight_initialization(mod_name, path)

    def emit_input(self, inp_name: str, path: Optional[os.PathLike] = None) -> str:
        """Emit C code to initialize input tensors."""
        return self._emit_input_initialization(inp_name, path)

    def _emit_operator_functions(self, mod_name: str) -> str:
        """Generate C function implementations for each operator type."""
        operator_functions = []

        # Common tensor operation functions with raw pointers
        operator_functions.append("""
// Tensor utility functions with raw pointers
int compute_tensor_size(const int* shape, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) {
        size *= shape[i];
    }
    return size;
}

int get_tensor_offset(const int* shape, const int* indices, int ndims) {
    int offset = 0;
    int stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        offset += indices[i] * stride;
        stride *= shape[i];
    }
    return offset;
}

float* allocate_tensor(const int* shape, int ndims) {
    int size = compute_tensor_size(shape, ndims);
    return (float*)malloc(size * sizeof(float));
}

void free_tensor(float* data) {
    if (data) {
        free(data);
    }
}

// Helper functions for common tensor shapes
int get_tensor_rank(float* tensor) {
    // For simplicity, we'll pass rank as a separate parameter
    // This function is just for compatibility
    return 1;
}

void get_tensor_shape(float* tensor, int* shape, int ndims) {
    // Shape will be passed separately as parameters
    // This function is just for compatibility
}
""")

        # Basic arithmetic operations
        operator_functions.append("""
// Addition operation: c = a + b (element-wise)
void op_add(const float* a, const float* b, float* c, int size) {
    // Assume broadcasting is handled by NNSmith to match shapes
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Subtraction operation: c = a - b (element-wise)
void op_sub(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] - b[i];
    }
}

// Multiplication operation: c = a * b (element-wise)
void op_mul(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

// Division operation: c = a / b (element-wise)
void op_div(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] / b[i];
    }
}
""")

        # Activation functions
        operator_functions.append("""
// ReLU activation
void op_relu(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

// Sigmoid activation
void op_sigmoid(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// Tanh activation
void op_tanh(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = tanhf(x[i]);
    }
}
""")

        # Reduction operations
        operator_functions.append("""
// Sum reduction along all dimensions
float op_sum(const float* x, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i];
    }
    return sum;
}

// Mean reduction along all dimensions
float op_mean(const float* x, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i];
    }
    return sum / size;
}
""")

        # Matrix multiplication (2D)
        operator_functions.append("""
// Matrix multiplication: C = A * B
void op_matmul(const float* a, const float* b, float* c, int M, int K, int N) {
    // Assume 2D matrices: a[M][K], b[K][N], c[M][N]
    // Initialize to zero
    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }

    // Matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                c[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }
}
""")

        # Convolution operations (simplified 2D)
        operator_functions.append("""
// 2D Convolution (simplified)
void op_conv2d(const float* input, const float* weight, const float* bias, float* output,
               int N, int H_in, int W_in, int C_in,
               int H_k, int W_k, int C_out,
               int stride_h, int stride_w, int pad_h, int pad_w) {
    // input: [N, H, W, C_in], weight: [H_k, W_k, C_in, C_out], bias: [C_out], output: [N, H_out, W_out, C_out]
    int H_out = (H_in + 2 * pad_h - H_k) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - W_k) / stride_w + 1;

    // Initialize to zero
    int output_size = N * H_out * W_out * C_out;
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
    }

    // Convolution computation
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int c_out = 0; c_out < C_out; c_out++) {
                    float sum = 0.0f;
                    for (int h_k = 0; h_k < H_k; h_k++) {
                        for (int w_k = 0; w_k < W_k; w_k++) {
                            for (int c_in = 0; c_in < C_in; c_in++) {
                                int h_in = h_out * stride_h - pad_h + h_k;
                                int w_in = w_out * stride_w - pad_w + w_k;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    int input_idx = ((n * H_in + h_in) * W_in + w_in) * C_in + c_in;
                                    int weight_idx = ((h_k * W_k + w_k) * C_in + c_in) * C_out + c_out;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                    int output_idx = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out;
                    output[output_idx] = sum + bias[c_out];
                }
            }
        }
    }
}
""")

        # Additional operations
        operator_functions.append("""
// Round operation
void op_round(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = roundf(x[i]);
    }
}

// Floor operation
void op_floor(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = floorf(x[i]);
    }
}

// Ceil operation
void op_ceil(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = ceilf(x[i]);
    }
}

// Absolute value
void op_abs(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = fabsf(x[i]);
    }
}

// ReduceMin along all dimensions
float op_reducemin(const float* x, int size) {
    if (size <= 0) return 0.0f;
    float min_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
        }
    }
    return min_val;
}

// ReduceMax along all dimensions
float op_reducemax(const float* x, int size) {
    if (size <= 0) return 0.0f;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    return max_val;
}

// Constant operation (initialize with value)
void op_constant(float* y, int size, float value) {
    for (int i = 0; i < size; i++) {
        y[i] = value;
    }
}

// Reshape operation (just copy data, shape is handled separately)
void op_reshape(const float* x, float* y, int size) {
    memcpy(y, x, size * sizeof(float));
}

// Expand operation (broadcast to larger tensor)
void op_expand(const float* x, float* y, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        y[i] = x[i % input_size];
    }
}

// Slice operation
void op_slice(const float* x, float* y, const int* input_shape, const int* output_shape,
              const int* start_indices, int ndims) {
    // Simplified slice - assumes contiguous memory
    int output_size = 1;
    for (int i = 0; i < ndims; i++) {
        output_size *= output_shape[i];
    }

    for (int i = 0; i < output_size; i++) {
        y[i] = x[i]; // Simplified - should calculate proper indices
    }
}

// Transpose operation (2D)
void op_transpose_2d(const float* x, float* y, int H, int W) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            y[j * H + i] = x[i * W + j];
        }
    }
}

// Element-wise minimum
void op_min(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (a[i] < b[i]) ? a[i] : b[i];
    }
}

// Element-wise maximum
void op_max(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

// Clip operation
void op_clip(const float* x, float* y, int size, float min_val, float max_val) {
    for (int i = 0; i < size; i++) {
        if (x[i] < min_val) {
            y[i] = min_val;
        } else if (x[i] > max_val) {
            y[i] = max_val;
        } else {
            y[i] = x[i];
        }
    }
}

// Generic transpose for N-dimensional tensors (simplified)
void op_transpose(const float* x, float* y, const int* input_shape, const int* perm, int ndims) {
    // Simplified: for now just copy data
    // Full implementation would need to calculate proper indices based on permutation
    int total_size = 1;
    for (int i = 0; i < ndims; i++) {
        total_size *= input_shape[i];
    }
    memcpy(y, x, total_size * sizeof(float));
}

// Atan operation
void op_atan(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = atanf(x[i]);
    }
}

// Cast to boolean (0 or 1)
void op_cast_bool(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = (x[i] != 0.0f) ? 1.0f : 0.0f;
    }
}

// Reflect padding (simplified - just copy input)
void op_reflect_pad(const float* x, float* y, int input_size, int output_size, const int* pads) {
    // Simplified implementation: just copy input to beginning of output
    memcpy(y, x, input_size * sizeof(float));
    // Zero out the padded regions
    for (int i = input_size; i < output_size; i++) {
        y[i] = 0.0f;
    }
}

// Expand in last 4 dimensions
void op_expand_last4(const float* x, float* y, int input_size, int output_size) {
    // Simplified: repeat input pattern
    for (int i = 0; i < output_size; i++) {
        y[i] = x[i % input_size];
    }
}

// Sin operation
void op_sin(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = sinf(x[i]);
    }
}

// Cos operation
void op_cos(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = cosf(x[i]);
    }
}

// Log operation
void op_log(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = logf(x[i]);
    }
}

// Exp operation
void op_exp(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = expf(x[i]);
    }
}

// Sqrt operation
void op_sqrt(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = sqrtf(x[i]);
    }
}

// Cast to int32 (simplified - just round to nearest int and back to float)
void op_cast_i32(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = (float)((int32_t)roundf(x[i]));
    }
}

// Triu (upper triangular matrix) - simplified: just copy input
void op_triu(const float* x, float* y, int rows, int cols) {
    // Simplified implementation: just copy the entire matrix
    // Full implementation would zero out lower triangular part
    for (int i = 0; i < rows * cols; i++) {
        y[i] = x[i];
    }
}

// Neg operation
void op_neg(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = -x[i];
    }
}

// Reciprocal operation
void op_reciprocal(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = (x[i] != 0.0f) ? 1.0f / x[i] : 0.0f;
    }
}

// Greater than operation
void op_greater(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }
}

// Less than operation
void op_less(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
    }
}

// Equal operation
void op_equal(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (fabsf(a[i] - b[i]) < 1e-6f) ? 1.0f : 0.0f;
    }
}

// Power operation
void op_pow(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = powf(a[i], b[i]);
    }
}

// GELU activation
void op_gelu(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = 0.5f * x[i] * (1.0f + tanhf(x[i] * 0.70710678f));  // Simplified GELU
    }
}

// Leaky ReLU
void op_leaky_relu(const float* x, float* y, int size, float negative_slope) {
    for (int i = 0; i < size; i++) {
        y[i] = (x[i] >= 0.0f) ? x[i] : negative_slope * x[i];
    }
}

// PReLU (parameterized ReLU)
void op_prelu(const float* x, const float* alpha, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = (x[i] >= 0.0f) ? x[i] : alpha[i] * x[i];
    }
}

// Softmax
void op_softmax(const float* x, float* y, int size, int axis) {
    // Simplified softmax on entire tensor
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }

    for (int i = 0; i < size; i++) {
        y[i] /= sum;
    }
}

// Asin operation
void op_asin(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = asinf(x[i]);
    }
}

// Acos operation
void op_acos(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = acosf(x[i]);
    }
}

// Tan operation
void op_tan(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = tanf(x[i]);
    }
}

// Log2 operation
void op_log2(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = log2f(x[i]);
    }
}

// MaxPool2d (simplified)
void op_maxpool2d(const float* x, float* y, int batch, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    // Very simplified implementation - just copy input to output
    int output_size = batch * channels * height * width;
    for (int i = 0; i < output_size; i++) {
        y[i] = x[i];
    }
}

// AvgPool2d (simplified)
void op_avgpool2d(const float* x, float* y, int batch, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    // Very simplified implementation - just copy input to output
    int output_size = batch * channels * height * width;
    for (int i = 0; i < output_size; i++) {
        y[i] = x[i];
    }
}

// Squeeze operation (remove dimensions of size 1)
void op_squeeze(const float* x, float* y, int input_size) {
    memcpy(y, x, input_size * sizeof(float));
}

// Unsqueeze operation (add dimension of size 1)
void op_unsqueeze(const float* x, float* y, int input_size) {
    memcpy(y, x, input_size * sizeof(float));
}

// ReduceProd
float op_reduceprod(const float* x, int size) {
    if (size <= 0) return 0.0f;
    float prod = 1.0f;
    for (int i = 0; i < size; i++) {
        prod *= x[i];
    }
    return prod;
}

// ArgMin (returns index of minimum value)
int op_argmin(const float* x, int size) {
    if (size <= 0) return 0;
    int min_idx = 0;
    float min_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
            min_idx = i;
        }
    }
    return min_idx;
}

// ArgMax (returns index of maximum value)
int op_argmax(const float* x, int size) {
    if (size <= 0) return 0;
    int max_idx = 0;
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// Cast to float32
void op_cast_f32(const float* x, float* y, int size) {
    memcpy(y, x, size * sizeof(float));
}

// Cast to float64 (simplified - just copy)
void op_cast_f64(const float* x, float* y, int size) {
    memcpy(y, x, size * sizeof(float));
}

// Cast to int64
void op_cast_i64(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = (float)((int64_t)x[i]);
    }
}

// Logical AND
void op_and(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = ((a[i] != 0.0f) && (b[i] != 0.0f)) ? 1.0f : 0.0f;
    }
}

// Logical OR
void op_or(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = ((a[i] != 0.0f) || (b[i] != 0.0f)) ? 1.0f : 0.0f;
    }
}

// Logical XOR
void op_xor(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        bool a_val = a[i] != 0.0f;
        bool b_val = b[i] != 0.0f;
        c[i] = (a_val ^ b_val) ? 1.0f : 0.0f;
    }
}

// Where operation (conditional selection)
void op_where(const float* condition, const float* x, const float* y, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (condition[i] != 0.0f) ? x[i] : y[i];
    }
}

// Concat operations (simplified - just concatenate)
void op_concat1(const float** inputs, float* output, const int* input_sizes, int num_inputs) {
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        memcpy(output + offset, inputs[i], input_sizes[i] * sizeof(float));
        offset += input_sizes[i];
    }
}

// BatchNorm2d (simplified - just copy input)
void op_batchnorm2d(const float* x, const float* gamma, const float* beta,
                     const float* mean, const float* var, float* y, int size) {
    // Simplified implementation: just copy input
    memcpy(y, x, size * sizeof(float));
}

// Interpolation operations (simplified - just copy)
void op_nearest_interp(const float* x, float* y, int input_size, int output_size) {
    memcpy(y, x, input_size * sizeof(float));
    // Zero out remaining elements if output is larger
    for (int i = input_size; i < output_size; i++) {
        y[i] = 0.0f;
    }
}

void op_linear_interp(const float* x, float* y, int input_size, int output_size) {
    op_nearest_interp(x, y, input_size, output_size);
}

void op_bilinear_interp(const float* x, float* y, int input_size, int output_size) {
    op_nearest_interp(x, y, input_size, output_size);
}

void op_bicubic_interp(const float* x, float* y, int input_size, int output_size) {
    op_nearest_interp(x, y, input_size, output_size);
}

void op_trilinear_interp(const float* x, float* y, int input_size, int output_size) {
    op_nearest_interp(x, y, input_size, output_size);
}

// Conv1d (simplified - just copy)
void op_conv1d(const float* input, const float* weight, const float* bias, float* output,
                int batch, int in_channels, int out_channels, int input_size, int kernel_size,
                int stride, int padding) {
    // Simplified: just copy input to output
    int output_size = batch * out_channels * input_size;
    memcpy(output, input, output_size * sizeof(float));
}

// NCHWConv2d (simplified - just copy)
void op_nchw_conv2d(const float* input, const float* weight, const float* bias, float* output,
                     int batch, int in_channels, int out_channels, int height, int width,
                     int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    // Simplified: just copy input to output
    int output_size = batch * out_channels * height * width;
    memcpy(output, input, output_size * sizeof(float));
}

// Tril (lower triangular)
void op_tril(const float* x, float* y, int rows, int cols) {
    // Simplified: just copy the entire matrix
    // Full implementation would zero out upper triangular part
    for (int i = 0; i < rows * cols; i++) {
        y[i] = x[i];
    }
}

// ConstPad (constant padding)
void op_const_pad(const float* x, float* y, int input_size, int output_size, float pad_value) {
    memcpy(y, x, input_size * sizeof(float));
    for (int i = input_size; i < output_size; i++) {
        y[i] = pad_value;
    }
}

// ReplicatePad
void op_replicate_pad(const float* x, float* y, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        y[i] = x[i < input_size ? i : input_size - 1];
    }
}

// Additional operator implementations for completeness

// ReduceL1 (L1 norm)
float op_reducel1(const float* x, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += fabsf(x[i]);
    }
    return sum;
}

// ReduceL2 (L2 norm)
float op_reducel2(const float* x, int size) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    return sqrtf(sum_sq);
}

// Sign operation
void op_sign(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] > 0.0f) {
            y[i] = 1.0f;
        } else if (x[i] < 0.0f) {
            y[i] = -1.0f;
        } else {
            y[i] = 0.0f;
        }
    }
}

// IsNan operation
void op_isnan(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = isnan(x[i]) ? 1.0f : 0.0f;
    }
}

// IsInf operation
void op_isinf(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = isinf(x[i]) ? 1.0f : 0.0f;
    }
}

// IsFinite operation
void op_isfinite(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = isfinite(x[i]) ? 1.0f : 0.0f;
    }
}

// Logical Not
void op_not(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = (x[i] == 0.0f) ? 1.0f : 0.0f;
    }
}

// GreaterThanOrEqual
void op_greater_equal(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (a[i] >= b[i]) ? 1.0f : 0.0f;
    }
}

// LessThanOrEqual
void op_less_equal(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (a[i] <= b[i]) ? 1.0f : 0.0f;
    }
}

// NotEqual
void op_not_equal(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (fabsf(a[i] - b[i]) >= 1e-6f) ? 1.0f : 0.0f;
    }
}

// Remainder operation
void op_remainder(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        if (b[i] != 0.0f) {
            c[i] = fmodf(a[i], b[i]);
        } else {
            c[i] = 0.0f;  // Handle division by zero
        }
    }
}

// FloorDivide operation
void op_floor_divide(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        if (b[i] != 0.0f) {
            c[i] = floorf(a[i] / b[i]);
        } else {
            c[i] = 0.0f;  // Handle division by zero
        }
    }
}

// Bitwise shift left (simplified for floats)
void op_left_shift(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        int shift = (int)b[i];
        if (shift >= 0 && shift < 32) {
            c[i] = (float)((int)a[i] << shift);
        } else {
            c[i] = 0.0f;
        }
    }
}

// Bitwise shift right (simplified for floats)
void op_right_shift(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        int shift = (int)b[i];
        if (shift >= 0 && shift < 32) {
            c[i] = (float)((int)a[i] >> shift);
        } else {
            c[i] = 0.0f;
        }
    }
}

// Bitwise AND (simplified for floats)
void op_bitwise_and(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (float)((int)a[i] & (int)b[i]);
    }
}

// Bitwise OR (simplified for floats)
void op_bitwise_or(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (float)((int)a[i] | (int)b[i]);
    }
}

// Bitwise XOR (simplified for floats)
void op_bitwise_xor(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (float)((int)a[i] ^ (int)b[i]);
    }
}

// Bitwise NOT (simplified for floats)
void op_bitwise_not(const float* a, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = (float)(~(int)a[i]);
    }
}

// Erf operation (Error function)
void op_erf(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = erff(x[i]);
    }
}

// Erfc operation (Complementary error function)
void op_erfc(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = erfcf(x[i]);
    }
}

// Log10 operation
void op_log10(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = log10f(x[i]);
    }
}

// Log1p operation (log(1 + x))
void op_log1p(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = log1pf(x[i]);
    }
}

// Expm1 operation (exp(x) - 1)
void op_expm1(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = expm1f(x[i]);
    }
}

// Square operation
void op_square(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = x[i] * x[i];
    }
}

// Cube operation
void op_cube(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = x[i] * x[i] * x[i];
    }
}

// Rsqrt operation (reciprocal square root)
void op_rsqrt(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] > 0.0f) {
            y[i] = 1.0f / sqrtf(x[i]);
        } else {
            y[i] = 0.0f;  // Handle non-positive input
        }
    }
}

// Softplus operation
void op_softplus(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] > 15.0f) {  // Avoid overflow
            y[i] = x[i];
        } else if (x[i] < -15.0f) {  // Avoid underflow
            y[i] = 0.0f;
        } else {
            y[i] = log1pf(expf(x[i]));
        }
    }
}

// Silu operation (Swish: x * sigmoid(x))
void op_silu(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * sigmoid_x;
    }
}

// Hardswish operation
void op_hardswish(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        float relu6_plus_3 = fmaxf(0.0f, fminf(6.0f, x[i] + 3.0f));
        y[i] = x[i] * relu6_plus_3 / 6.0f;
    }
}

// Mish operation
void op_mish(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        float softplus_x = (x[i] > 15.0f) ? x[i] : ((x[i] < -15.0f) ? 0.0f : log1pf(expf(x[i])));
        float tanh_softplus = tanhf(softplus_x);
        y[i] = x[i] * tanh_softplus;
    }
}

// Hardtanh operation
void op_hardtanh(const float* x, float* y, int size, float min_val, float max_val) {
    for (int i = 0; i < size; i++) {
        if (x[i] < min_val) {
            y[i] = min_val;
        } else if (x[i] > max_val) {
            y[i] = max_val;
        } else {
            y[i] = x[i];
        }
    }
}

// Hardshrink operation
void op_hardshrink(const float* x, float* y, int size, float lambd) {
    for (int i = 0; i < size; i++) {
        if (x[i] > lambd || x[i] < -lambd) {
            y[i] = x[i];
        } else {
            y[i] = 0.0f;
        }
    }
}

// Softshrink operation
void op_softshrink(const float* x, float* y, int size, float lambd) {
    for (int i = 0; i < size; i++) {
        if (x[i] > lambd) {
            y[i] = x[i] - lambd;
        } else if (x[i] < -lambd) {
            y[i] = x[i] + lambd;
        } else {
            y[i] = 0.0f;
        }
    }
}

// Relu6 operation
void op_relu6(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        y[i] = fmaxf(0.0f, fminf(6.0f, x[i]));
    }
}

// Elu operation
void op_elu(const float* x, float* y, int size, float alpha) {
    for (int i = 0; i < size; i++) {
        if (x[i] >= 0.0f) {
            y[i] = x[i];
        } else {
            y[i] = alpha * (expf(x[i]) - 1.0f);
        }
    }
}

// Celu operation
void op_celu(const float* x, float* y, int size, float alpha) {
    for (int i = 0; i < size; i++) {
        if (x[i] >= 0.0f) {
            y[i] = x[i];
        } else {
            y[i] = alpha * (expf(x[i] / alpha) - 1.0f);
        }
    }
}

// Selu operation
void op_selu(const float* x, float* y, int size) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;

    for (int i = 0; i < size; i++) {
        if (x[i] >= 0.0f) {
            y[i] = scale * x[i];
        } else {
            y[i] = scale * alpha * (expf(x[i]) - 1.0f);
        }
    }
}

// Glu operation (Gated Linear Unit)
void op_glu(const float* x, float* y, int size, int dim) {
    // Simplified GLU: splits input along dim and multiplies
    for (int i = 0; i < size / 2; i++) {
        float gate = 1.0f / (1.0f + expf(-x[i + size / 2]));
        y[i] = x[i] * gate;
    }
}

// HardSigmoid operation
void op_hardsigmoid(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        float val = (x[i] + 3.0f) / 6.0f;
        y[i] = fmaxf(0.0f, fminf(1.0f, val));
    }
}

// LogSigmoid operation
void op_logsigmoid(const float* x, float* y, int size) {
    for (int i = 0; i < size; i++) {
        // For numerical stability
        if (x[i] > 0.0f) {
            y[i] = -log1pf(expf(-x[i]));
        } else {
            y[i] = x[i] - log1pf(expf(x[i]));
        }
    }
}

// Softmin operation
void op_softmin(const float* x, float* y, int size, int axis) {
    // Softmin = softmax(-x)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        y[i] = expf(-(x[i] - max_val));
        sum += y[i];
    }

    for (int i = 0; i < size; i++) {
        y[i] /= sum;
    }
}

// LogSoftmax operation
void op_logsoftmax(const float* x, float* y, int size, int axis) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }

    float log_sum = logf(sum);
    for (int i = 0; i < size; i++) {
        y[i] = x[i] - max_val - log_sum;
    }
}
""")

        # Main model function declaration
        operator_functions.append(f"""
// Main model function declaration
void {mod_name}_forward(float** inputs, float** outputs,
                       const int** input_shapes, const int** output_shapes,
                       const int* input_ndims, const int* output_ndims);
""")

        return "\n".join(operator_functions)

    def _emit_graph_execution(self, out_name: str, mod_name: str, inp_name: str) -> str:
        """Generate the main computational graph execution code by traversing GraphIR."""
        execution_code = []

        # Count inputs and outputs
        num_inputs = len(self.input_like)
        num_outputs = len(self.output_like)

        # Map variable names to tensor indices
        var_to_tensor_idx = {}
        tensor_counter = 0
        variable_declarations = []
        shape_declarations = []

        execution_code.append(f"""
// Graph execution for {mod_name}
void {mod_name}_forward(float** inputs, float** outputs,
                       const int** input_shapes, const int** output_shapes,
                       const int* input_ndims, const int* output_ndims) {{
    // inputs[0..{num_inputs-1}]: input tensor data (raw pointers)
    // outputs[0..{num_outputs-1}]: output tensor data (raw pointers)
    // Shape information is passed separately
""")

        # Add input mapping with individual variable declarations
        input_idx = 0
        for inst in self.ir.insts:
            if hasattr(inst.iexpr.op, '__class__') and inst.iexpr.op.__class__.__name__ == 'Input':
                var_name = f"v_{tensor_counter}"  # Simple v_0, v_1, ... naming
                shape_name = f"v_{tensor_counter}_shape"
                ndim_name = f"v_{tensor_counter}_ndim"

                # Add variable declarations
                variable_declarations.append(f"    float* {var_name};")
                shape_declarations.append(f"    int {shape_name}[4];")
                shape_declarations.append(f"    int {ndim_name};")

                execution_code.append(f"""
    // Input {var_name}
    {var_name} = inputs[{input_idx}];
    {shape_name}[0] = {self.ir.vars[inst.retval()].shape[0] if len(self.ir.vars[inst.retval()].shape) > 0 else 1};
    {shape_name}[1] = {self.ir.vars[inst.retval()].shape[1] if len(self.ir.vars[inst.retval()].shape) > 1 else 1};
    {shape_name}[2] = {self.ir.vars[inst.retval()].shape[2] if len(self.ir.vars[inst.retval()].shape) > 2 else 1};
    {shape_name}[3] = {self.ir.vars[inst.retval()].shape[3] if len(self.ir.vars[inst.retval()].shape) > 3 else 1};
    {ndim_name} = {len(self.ir.vars[inst.retval()].shape)};
""")
                var_to_tensor_idx[inst.retval()] = tensor_counter
                input_idx += 1
                tensor_counter += 1

        # Generate computation for each instruction
        for inst in self.ir.insts:
            op_name = inst.iexpr.op.__class__.__name__

            # Skip Input operations as they're handled above
            if op_name == 'Input':
                continue

            # Get input tensor indices
            input_tensor_indices = []
            for arg in inst.iexpr.args:
                if arg in var_to_tensor_idx:
                    input_tensor_indices.append(var_to_tensor_idx[arg])
                else:
                    # This shouldn't happen in a well-formed graph, but handle gracefully
                    execution_code.append(f"    // WARNING: Variable {arg} not found, using -1 as placeholder\n")
                    input_tensor_indices.append(-1)

            # Get output tensor info
            output_var = f"v_{tensor_counter}"  # Simple v_0, v_1, ... naming
            output_shape_name = f"v_{tensor_counter}_shape"
            output_ndim_name = f"v_{tensor_counter}_ndim"
            output_shape = self.ir.vars[inst.retval()].shape
            output_size = np.prod(output_shape)
            output_ndims = len(output_shape)

            # Add variable declarations for this tensor
            variable_declarations.append(f"    float* {output_var};")
            shape_declarations.append(f"    int {output_shape_name}[4];")
            shape_declarations.append(f"    int {output_ndim_name};")

            # Allocate output tensor
            execution_code.append(f"""
    // Allocate {output_var} (operation: {op_name})
    {output_var} = (float*)malloc({output_size} * sizeof(float));
    {output_shape_name}[0] = {output_shape[0] if len(output_shape) > 0 else 1};
    {output_shape_name}[1] = {output_shape[1] if len(output_shape) > 1 else 1};
    {output_shape_name}[2] = {output_shape[2] if len(output_shape) > 2 else 1};
    {output_shape_name}[3] = {output_shape[3] if len(output_shape) > 3 else 1};
    {output_ndim_name} = {output_ndims};
""")

            # Generate operation-specific code using individual variables
            if len(input_tensor_indices) > 0 and all(idx >= 0 for idx in input_tensor_indices):
                first_input_idx = input_tensor_indices[0]
                first_input_var = f"v_{first_input_idx}"

                if op_name == 'Add':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_add({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Sub':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_sub({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Mul':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_mul({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Div':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_div({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Pow':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_pow({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'ReLU':
                    execution_code.append(f"    op_relu({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Sigmoid':
                    execution_code.append(f"    op_sigmoid({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Tanh':
                    execution_code.append(f"    op_tanh({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'GELU':
                    execution_code.append(f"    op_gelu({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'LeakyReLU':
                    # Default negative slope if not specified
                    execution_code.append(f"    op_leaky_relu({first_input_var}, {output_var}, {output_size}, 0.01f);\n")

                elif op_name == 'PReLU':
                    # Use first input as x, second as alpha
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_prelu({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")
                    else:
                        execution_code.append(f"    op_leaky_relu({first_input_var}, {output_var}, {output_size}, 0.01f);\n")

                elif op_name == 'Softmax':
                    # Default axis = -1 (last dimension)
                    execution_code.append(f"    op_softmax({first_input_var}, {output_var}, {output_size}, -1);\n")

                elif op_name == 'Sin':
                    execution_code.append(f"    op_sin({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Cos':
                    execution_code.append(f"    op_cos({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Asin':
                    execution_code.append(f"    op_asin({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Acos':
                    execution_code.append(f"    op_acos({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Tan':
                    execution_code.append(f"    op_tan({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Atan':
                    execution_code.append(f"    op_atan({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Abs':
                    execution_code.append(f"    op_abs({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Neg':
                    execution_code.append(f"    op_neg({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Reciprocal':
                    execution_code.append(f"    op_reciprocal({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Round':
                    execution_code.append(f"    op_round({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Floor':
                    execution_code.append(f"    op_floor({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Ceil':
                    execution_code.append(f"    op_ceil({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Sqrt':
                    execution_code.append(f"    op_sqrt({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Log2':
                    execution_code.append(f"    op_log2({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Log':
                    execution_code.append(f"    op_log({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Exp':
                    execution_code.append(f"    op_exp({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'CastF32':
                    execution_code.append(f"    op_cast_f32({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'CastF64':
                    execution_code.append(f"    op_cast_f64({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'CastI32':
                    execution_code.append(f"    op_cast_i32({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'CastI64':
                    execution_code.append(f"    op_cast_i64({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'CastBool':
                    execution_code.append(f"    op_cast_bool({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Min':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_min({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Max':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_max({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Clip':
                    # Default clip values if not specified
                    execution_code.append(f"    op_clip({first_input_var}, {output_var}, {output_size}, 0.0f, 1.0f);\n")

                elif op_name == 'Greater':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_greater({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Less':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_less({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Equal':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_equal({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'And':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_and({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Or':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_or({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Xor':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        execution_code.append(f"    op_xor({first_input_var}, {second_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Where':
                    if len(input_tensor_indices) >= 3:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        third_input_var = f"v_{input_tensor_indices[2]}"
                        execution_code.append(f"    op_where({first_input_var}, {second_input_var}, {third_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'MatMul' or op_name == 'PTMatMul':
                    if len(input_tensor_indices) >= 2:
                        second_input_var = f"v_{input_tensor_indices[1]}"
                        # Simplified matrix multiplication - assume 2D
                        if len(output_shape) >= 2:
                            M, K, N = output_shape[0], output_shape[1], output_shape[1]  # Simplified
                            execution_code.append(f"    op_matmul({first_input_var}, {second_input_var}, {output_var}, {M}, {K}, {N});\n")
                        else:
                            execution_code.append(f"    // MatMul: insufficient shape info, copying input\n")
                            execution_code.append(f"    memcpy({output_var}, {first_input_var}, {output_size} * sizeof(float));\n")

                elif op_name == 'NCHWConv2d':
                    # Simplified 2D convolution
                    execution_code.append(f"    // NCHWConv2d: using simplified implementation\n")
                    execution_code.append(f"    op_nchw_conv2d({first_input_var}, NULL, NULL, {output_var}, 1, 3, 3, {output_shape[2]}, {output_shape[3]}, 3, 3, 1, 1, 0, 0);\n")

                elif op_name == 'Conv1d':
                    # Simplified 1D convolution
                    execution_code.append(f"    // Conv1d: using simplified implementation\n")
                    execution_code.append(f"    op_conv1d({first_input_var}, NULL, NULL, {output_var}, 1, 3, 3, {output_shape[1] if len(output_shape) > 1 else 1}, 3, 1, 0);\n")

                elif op_name == 'Conv2d':
                    # Simplified 2D convolution
                    execution_code.append(f"    // Conv2d: using simplified implementation\n")
                    execution_code.append(f"    op_conv2d({first_input_var}, NULL, NULL, {output_var}, 1, {output_shape[2]}, {output_shape[3]}, 3, 3, 3, 1, 1, 0, 0);\n")

                elif op_name == 'MaxPool2d':
                    execution_code.append(f"    // MaxPool2d: using simplified implementation\n")
                    execution_code.append(f"    op_maxpool2d({first_input_var}, {output_var}, 1, {output_shape[1] if len(output_shape) > 1 else 1}, {output_shape[2] if len(output_shape) > 2 else 1}, {output_shape[3] if len(output_shape) > 3 else 1}, 2, 2, 2, 2, 0, 0);\n")

                elif op_name == 'AvgPool2d':
                    execution_code.append(f"    // AvgPool2d: using simplified implementation\n")
                    execution_code.append(f"    op_avgpool2d({first_input_var}, {output_var}, 1, {output_shape[1] if len(output_shape) > 1 else 1}, {output_shape[2] if len(output_shape) > 2 else 1}, {output_shape[3] if len(output_shape) > 3 else 1}, 2, 2, 2, 2, 0, 0);\n")

                elif op_name == 'BatchNorm2d':
                    execution_code.append(f"    // BatchNorm2d: using simplified implementation\n")
                    execution_code.append(f"    op_batchnorm2d({first_input_var}, NULL, NULL, NULL, NULL, {output_var}, {output_size});\n")

                elif op_name == 'ExpandLast4':
                    execution_code.append(f"    op_expand_last4({first_input_var}, {output_var}, {output_size // 8}, {output_size});\n")

                elif op_name == 'Expand' or op_name == 'ExpandLast1' or op_name == 'ExpandLast2' or op_name == 'ExpandLast3':
                    execution_code.append(f"    op_expand({first_input_var}, {output_var}, {output_size // 2}, {output_size});\n")

                elif op_name == 'Reshape':
                    execution_code.append(f"    op_reshape({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Transpose':
                    execution_code.append(f"    // Transpose: using simplified implementation\n")
                    execution_code.append(f"    op_transpose({first_input_var}, {output_var}, (int[]){{1, 1, 1, 1}}, (int[]){{0, 1, 2, 3}}, {len(output_shape)});\n")

                elif op_name == 'Slice':
                    execution_code.append(f"    // Slice: using simplified implementation\n")
                    execution_code.append(f"    op_slice({first_input_var}, {output_var}, (int[]){{1, 1}}, (int[]){{1, 1}}, (int[]){{0, 0}}, {len(output_shape)});\n")

                elif op_name == 'ConstPad':
                    execution_code.append(f"    op_const_pad({first_input_var}, {output_var}, {output_size // 2}, {output_size}, 0.0f);\n")

                elif op_name == 'ReplicatePad':
                    execution_code.append(f"    op_replicate_pad({first_input_var}, {output_var}, {output_size // 2}, {output_size});\n")

                elif op_name == 'ReflectPad':
                    execution_code.append(f"    op_reflect_pad({first_input_var}, {output_var}, {output_size // 2}, {output_size}, (int[]){{1, 1}});\n")

                elif op_name == 'Squeeze':
                    execution_code.append(f"    op_squeeze({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Unsqueeze':
                    execution_code.append(f"    op_unsqueeze({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Concat1' or op_name == 'Concat2' or op_name == 'Concat3' or op_name == 'Concat4' or op_name == 'Concat5':
                    execution_code.append(f"    // Concat{op_name[-1]}: using simplified implementation\n")
                    execution_code.append(f"    op_concat1((const float*[]){{{first_input_var}}}, {output_var}, (int[]){{{output_size}}}, 1);\n")

                elif op_name == 'NearestInterp':
                    execution_code.append(f"    op_nearest_interp({first_input_var}, {output_var}, {output_size}, {output_size});\n")

                elif op_name == 'LinearInterp':
                    execution_code.append(f"    op_linear_interp({first_input_var}, {output_var}, {output_size}, {output_size});\n")

                elif op_name == 'BilinearInterp':
                    execution_code.append(f"    op_bilinear_interp({first_input_var}, {output_var}, {output_size}, {output_size});\n")

                elif op_name == 'BicubicInterp':
                    execution_code.append(f"    op_bicubic_interp({first_input_var}, {output_var}, {output_size}, {output_size});\n")

                elif op_name == 'TrilinearInterp':
                    execution_code.append(f"    op_trilinear_interp({first_input_var}, {output_var}, {output_size}, {output_size});\n")

                elif op_name == 'TorchReduceSum':
                    # Return scalar result
                    sum_val = f"op_sum({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = {sum_val};\n")

                elif op_name == 'ReduceSum':
                    # Return scalar result
                    sum_val = f"op_sum({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = {sum_val};\n")

                elif op_name == 'ReduceMin':
                    # Return scalar result
                    min_val = f"op_reducemin({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = {min_val};\n")

                elif op_name == 'ReduceMax':
                    # Return scalar result
                    max_val = f"op_reducemax({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = {max_val};\n")

                elif op_name == 'ReduceMean':
                    # Return scalar result
                    mean_val = f"op_mean({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = {mean_val};\n")

                elif op_name == 'ReduceProd':
                    # Return scalar result
                    prod_val = f"op_reduceprod({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = {prod_val};\n")

                elif op_name == 'ArgMin':
                    # Return scalar result (index)
                    argmin_val = f"op_argmin({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = (float){argmin_val};\n")

                elif op_name == 'ArgMax':
                    # Return scalar result (index)
                    argmax_val = f"op_argmax({first_input_var}, {output_size})"
                    execution_code.append(f"    {output_var}[0] = (float){argmax_val};\n")

                elif op_name == 'Triu':
                    if len(output_shape) >= 2:
                        rows, cols = output_shape[0], output_shape[1]
                        execution_code.append(f"    op_triu({first_input_var}, {output_var}, {rows}, {cols});\n")
                    else:
                        execution_code.append(f"    op_reshape({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Tril':
                    if len(output_shape) >= 2:
                        rows, cols = output_shape[0], output_shape[1]
                        execution_code.append(f"    op_tril({first_input_var}, {output_var}, {rows}, {cols});\n")
                    else:
                        execution_code.append(f"    op_reshape({first_input_var}, {output_var}, {output_size});\n")

                elif op_name == 'Constant':
                    # Initialize with constant value (default 0.0)
                    execution_code.append(f"    op_constant({output_var}, {output_size}, 0.0f);\n")

                else:
                    # Fallback: just initialize with zeros or copy first input if available
                    if len(input_tensor_indices) > 0 and input_tensor_indices[0] >= 0:
                        execution_code.append(f"    // Unsupported operation: {op_name}, copying input to {output_var}\n")
                        execution_code.append(f"    memcpy({output_var}, {first_input_var}, {output_size} * sizeof(float));\n")
                    else:
                        execution_code.append(f"    // Unsupported operation: {op_name}, initializing {output_var} with zeros\n")
                        execution_code.append(f"    for (int i = 0; i < {output_size}; i++) {{ {output_var}[i] = 0.0f; }}\n")

            # Map output variable to tensor index
            var_to_tensor_idx[inst.retval()] = tensor_counter
            tensor_counter += 1

        # Update the variable declarations in the execution code
        current_execution = "\n".join(execution_code)

        # Rebuild execution code with proper variable declarations at the beginning
        final_execution_code = []

        # Add function header
        final_execution_code.append(f"""
// Graph execution for {mod_name}
void {mod_name}_forward(float** inputs, float** outputs,
                       const int** input_shapes, const int** output_shapes,
                       const int* input_ndims, const int* output_ndims) {{
    // inputs[0..{num_inputs-1}]: input tensor data (raw pointers)
    // outputs[0..{num_outputs-1}]: output tensor data (raw pointers)
    // Shape information is passed separately

    // Variable declarations for intermediate tensors
""")

        # Add all variable declarations
        if variable_declarations:
            final_execution_code.extend(variable_declarations)
        if shape_declarations:
            final_execution_code.extend(shape_declarations)

        # Add the rest of the execution code (without the function header)
        rest_of_code = current_execution.split("{", 1)[1]  # Split after the first {
        final_execution_code.append(rest_of_code)

        # Replace array references with individual variables in the final code
        final_code_str = "\n".join(final_execution_code)

        # Copy outputs to output buffers using individual variables
        output_idx = 0
        for var_name, tensor_spec in self.output_like.items():
            if var_name in var_to_tensor_idx:
                tensor_idx = var_to_tensor_idx[var_name]
                output_var = f"v_{tensor_idx}"
                output_size = np.prod(tensor_spec.shape)
                final_code_str += f"""
    // Copy output {var_name} to output buffer {output_idx}
    memcpy(outputs[{output_idx}], {output_var}, {output_size} * sizeof(float));
"""
                output_idx += 1

        # Clean up intermediate tensors (excluding input tensors)
        input_count = len(self.input_like)
        cleanup_code = f"""
    // Clean up intermediate tensors (excluding input tensors)
"""

        # Generate explicit cleanup for each variable
        for i in range(input_count, tensor_counter):
            var_to_clean = f"v_{i}"
            cleanup_code += f"    if ({var_to_clean} != NULL) {{ free({var_to_clean}); }}\n"

        final_code_str += cleanup_code + "}\n"

        # Helper function to run the model
        final_code_str += f"""
// Helper function to run the complete model
void run_{mod_name}(float** input_data, const int** input_shapes, const int* input_ndims,
                   float** output_data, const int** output_shapes, int* output_ndims) {{
    // Allocate output tensors
    for (int i = 0; i < {num_outputs}; i++) {{
        int output_size = compute_tensor_size(output_shapes[i], output_ndims[i]);
        output_data[i] = allocate_tensor(output_shapes[i], output_ndims[i]);
    }}

    // Run forward pass
    {mod_name}_forward(input_data, output_data, input_shapes, output_shapes, input_ndims, output_ndims);
}}
"""

        return final_code_str

    def _emit_weight_initialization(self, mod_name: str, path: Optional[os.PathLike] = None) -> str:
        """Generate C code to initialize model weights."""
        weight_code = []

        weight_code.append(f"""
// Weight initialization for {mod_name}
void initialize_{mod_name}_weights() {{
    // In practice, this would load weights from a file or initialize them

    // Example weight initialization
    // float conv_weight_data[] = {{...}};
    // float conv_bias_data[] = {{...}};

    // Load weights into the model
    // This would be specific to your model architecture
}}
""")

        return "\n".join(weight_code)

    def _emit_input_initialization(self, inp_name: str, path: Optional[os.PathLike] = None) -> str:
        """Generate C code to initialize input tensors."""
        input_code = []

        input_code.append(f"""
// Input initialization for testing
void initialize_test_inputs(float** input_data, const int** input_shapes, int* input_ndims) {{

    // Static shape arrays (could be made dynamic)
""")

        # Generate static shape arrays
        shape_arrays = []
        for i, (name, tensor_spec) in enumerate(self.input_like.items()):
            shape_str = ", ".join(map(str, tensor_spec.shape))
            shape_arrays.append(f"static int input_shape_{i}[] = {{{shape_str}}};")

        input_code.extend(shape_arrays)
        input_code.append("\n")

        # Initialize inputs
        for i, (name, tensor_spec) in enumerate(self.input_like.items()):
            shape_str = "x".join(map(str, tensor_spec.shape))
            size = np.prod(tensor_spec.shape)

            input_code.append(f"""
    // Input {i}: {name} (shape: {shape_str})
    input_shapes[{i}] = input_shape_{i};
    input_ndims[{i}] = {len(tensor_spec.shape)};
    input_data[{i}] = (float*)malloc({size} * sizeof(float));

    // Initialize with random values or test data
    for (int j = 0; j < {size}; j++) {{
        input_data[{i}][j] = (float)rand() / RAND_MAX;  // Random [0, 1]
    }}
""")

        input_code.append("""
}
""")

        return "\n".join(input_code)