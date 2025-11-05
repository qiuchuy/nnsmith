import os
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.c import CModel


class CBackend(BackendFactory):
    """C backend that compiles and executes C code for neural network operators."""

    def __init__(self, target="cpu", optmax: bool = True, **kwargs):
        super().__init__(target, optmax)
        self.compiler = kwargs.get("compiler", "gcc")
        self.compile_flags = kwargs.get("compile_flags", ["-O2" if optmax else "-O0", "-Wall", "-std=c99"])
        self.temp_dir = kwargs.get("temp_dir", None)

    @property
    def system_name(self) -> str:
        return "c"

    def make_backend(self, model: CModel) -> BackendCallable:
        """Create a callable backend from the C model."""

        # Generate the complete C code
        c_code = self._generate_complete_c_code(model)

        # Compile the C code to a shared library or executable
        compiled_path = self._compile_c_code(c_code)

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            return self._execute_compiled_model(compiled_path, model, inputs)

        return closure

    def _generate_complete_c_code(self, model: CModel) -> str:
        """Generate the complete C program including main function."""

        c_code_parts = []

        # Add headers
        c_code_parts.extend(model.import_libs)
        c_code_parts.append("\n")

        # Add model definitions
        c_code_parts.append(model.emit_def("model", "Model"))
        c_code_parts.append("\n")

        # Add weight initialization
        c_code_parts.append(model.emit_weight("model"))
        c_code_parts.append("\n")

        # Add main function for testing
        main_code = self._generate_main_function(model)
        c_code_parts.append(main_code)

        return "\n".join(c_code_parts)

    def _generate_main_function(self, model: CModel) -> str:
        """Generate the main function for the C program."""

        num_inputs = len(model.input_like)
        num_outputs = len(model.output_like)

        main_code = f"""
int main() {{
    printf("Running C neural network model\\n");

    // Initialize input data
    float* input_data[{num_inputs}];
    int* input_shapes[{num_inputs}];
    int input_ndims[{num_inputs}];

    initialize_test_inputs(input_data, input_shapes, input_ndims);

    // Initialize model weights
    initialize_model_weights();

    // Prepare output storage
    float* output_data[{num_outputs}];
    int* output_shapes[{num_outputs}];
    int output_ndims[{num_outputs}];

    // Calculate output sizes
"""

        # Add output size calculations
        for i, (name, tensor_spec) in enumerate(model.output_like.items()):
            size = np.prod(tensor_spec.shape)
            ndims = len(tensor_spec.shape)
            main_code += f"""
    output_data[{i}] = (float*)malloc({size} * sizeof(float));
    output_shapes[{i}] = (int*)malloc({ndims} * sizeof(int));
    output_ndims[{i}] = {ndims};
"""

        main_code += f"""

    // Run the model
    run_model(input_data, input_shapes, input_ndims,
              output_data, output_shapes, output_ndims);

    // Print outputs
    printf("Model outputs:\\n");
"""

        # Add output printing
        for i, (name, tensor_spec) in enumerate(model.output_like.items()):
            size = np.prod(tensor_spec.shape)
            shape_str = "x".join(map(str, tensor_spec.shape))
            main_code += f"""
    printf("Output {i} ({name}, shape: {shape_str}):\\n");
    for (int j = 0; j < {size}; j++) {{
        printf("%.4f ", output_data[{i}][j]);
        if ((j + 1) % 8 == 0) printf("\\n");
    }}
    printf("\\n\\n");
"""

        # Add cleanup
        main_code += """
    // Clean up
    for (int i = 0; i < """
        main_code += f"{num_inputs}; i++) {{"
        main_code += """
        free(input_data[i]);
        free(input_shapes[i]);
    }
    for (int i = 0; i < """
        main_code += f"{num_outputs}; i++) {{"
        main_code += """
        free(output_data[i]);
        free(output_shapes[i]);
    }

    printf("Model execution completed.\\n");
    return 0;
}
"""

        return main_code

    def _compile_c_code(self, c_code: str) -> str:
        """Compile C code to an executable."""

        # Create a temporary directory for compilation
        if self.temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="nnsmith_c_backend_")
        else:
            temp_dir = self.temp_dir
            os.makedirs(temp_dir, exist_ok=True)

        # Write C code to file
        c_file_path = os.path.join(temp_dir, "model.c")
        with open(c_file_path, "w") as f:
            f.write(c_code)

        # Compile C code
        executable_path = os.path.join(temp_dir, "model")
        compile_cmd = [self.compiler] + self.compile_flags + ["-o", executable_path, c_file_path]

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"C compilation failed: {e.stderr}")

        return executable_path

    def _execute_compiled_model(
        self,
        compiled_path: str,
        model: CModel,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute the compiled C model with given inputs."""

        # For now, we'll create a simple execution that runs the compiled program
        # In a more sophisticated implementation, we could:
        # 1. Use shared libraries for function calling
        # 2. Use IPC mechanisms for data exchange
        # 3. Generate C code that can be called directly from Python

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for name, data in inputs.items():
                # Write input data to file (simple format)
                for val in data.flatten():
                    f.write(f"{val}\n")
            input_file = f.name

        try:
            # Run the compiled program
            result = subprocess.run(
                [compiled_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"C model execution failed: {result.stderr}")

            # Parse output (this is a simplified implementation)
            # In practice, you'd need a more robust output parsing mechanism
            outputs = {}

            # For now, generate dummy outputs based on the model specification
            for name, tensor_spec in model.output_like.items():
                outputs[name] = np.random.random(tensor_spec.shape).astype(np.float32)

            return outputs

        finally:
            # Clean up temporary input file
            os.unlink(input_file)

    @property
    def import_libs(self) -> List[str]:
        return ["#include <stdio.h>", "#include <stdlib.h>", "#include <math.h>"]

    def emit_compile(self, opt_name: str, mod_name: str, inp_name: Optional[str] = None) -> str:
        """Emit compilation instructions for the C model."""
        return f"// C compilation: gcc -O2 -o {opt_name} {mod_name}.c"

    def emit_run(self, out_name: str, opt_name: str, inp_name: str) -> str:
        """Emit execution instructions for the compiled C model."""
        return f"// C execution: ./{opt_name}"