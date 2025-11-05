# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NNSmith is a Deep Neural Network (DNN) fuzzing and testing infrastructure that automatically generates random but valid neural networks to find bugs in deep learning frameworks and compilers. The project uses symbolic execution and constraint solving (Z3) to generate valid DNN models across multiple frameworks.

## Common Development Commands

### Installation
```shell
# Install with specific backends
pip install "nnsmith[torch,onnx,tvm,onnxruntime]" --upgrade
# Or install from HEAD
pip install "git+https://github.com/ise-uiuc/nnsmith@main#egg=nnsmith[torch,onnx,tvm,onnxruntime]" --upgrade

# Development dependencies
pip install -r requirements/dev.txt
pre-commit install
```

### Model Generation
```shell
# Generate random model (default: 5 nodes, ONNX format)
nnsmith.model_gen model.type=onnx debug.viz=true

# Generate PyTorch model with specific constraints
nnsmith.model_gen model.type=torch mgen.max_nodes=10 mgen.rank_choices="[4]" mgen.dtype_choices="[f32]"

# Generate with specific operators only
nnsmith.model_gen model.type=torch mgen.include="[core.NCHWConv2d, core.ReLU]" debug.viz=true
```

### Model Execution and Testing
```shell
# Execute and validate a model
nnsmith.model_exec model.type=onnx backend.type=onnxruntime model.path=nnsmith_output/model.onnx

# Cross-framework comparison testing
nnsmith.model_exec model.type=onnx backend.type=onnxruntime model.path=nnsmith_output/model.onnx cmp.with='{type:tvm, optmax:true, target:cpu}'
```

### Data Type Testing
```shell
# Infer operator support for backend
nnsmith.dtype_test model.type=onnx backend.type=onnxruntime
```

### Fuzzing
```shell
# Run fuzzing campaign
nnsmith.fuzz fuzz.time=300 model.type=onnx backend.type=tvm fuzz.root=fuzz_report debug.viz=true
```

### Testing
```shell
# Run backend-specific tests (environments may conflict)
pytest tests/core -s
pytest tests/torch -s
pytest tests/onnxruntime -s
pytest tests/tvm -s
pytest tests/tensorrt -s
```

## Architecture Overview

### Core Components

1. **Abstract Layer** (`nnsmith/abstract/`): Defines symbolic neural network operations with mathematical constraints using Z3 solver
   - `op.py`: Abstract operation definitions with constraints
   - `dtype.py`: Tensor data type system
   - `arith.py`: Mathematical constraint operations

2. **Graph Generation** (`nnsmith/graph_gen.py`, `nnsmith/gir.py`):
   - Symbolic graph generation using SMT solving
   - Graph Intermediate Representation (GIR) for model abstraction
   - Constraint satisfaction for valid network topologies

3. **Backend Layer** (`nnsmith/backends/`): Framework-specific implementations
   - Factory pattern for backend selection
   - Supports PyTorch, ONNX, TensorFlow, TVM, TensorRT, XLA, TFLite
   - Each backend handles model compilation and execution

4. **Materialization Layer** (`nnsmith/materialize/`): Converts abstract graphs to concrete models
   - Model execution across frameworks
   - Test case generation and validation
   - Bug reporting and oracle creation

5. **CLI Interface** (`nnsmith/cli/`): Command-line tools for all major operations

### Configuration System

Uses Hydra for configuration management (`nnsmith/config/main.yaml`):
- `model.*`: Model type and path settings
- `mgen.*`: Model generation parameters (nodes, timeout, constraints)
- `backend.*`: Backend configuration (optimization, target)
- `debug.*`: Debugging options (visualization)
- `fuzz.*`: Fuzzing campaign settings
- `cmp.*`: Cross-framework comparison settings

### Key Design Patterns

- **Abstract Operations**: Symbolic operator definitions with constraint-based validation
- **Factory Pattern**: Backend selection and initialization
- **Symbolic Execution**: Z3 SMT solver for generating valid neural network topologies
- **Hydra Configuration**: Flexible, hierarchical configuration management

## Development Guidelines

### Adding New Operators

1. Define abstract operator in `nnsmith/abstract/op.py`
2. Implement `__init__`, `type_transfer`, and `requires` methods
3. Specify input/output rank constraints and data types
4. Add constraints for valid parameter ranges
5. Implement backend-specific materializations

### Adding New Backends

1. Create backend implementation in `nnsmith/backends/`
2. Implement `BackendFactory` and `Model` interfaces
3. Add backend to factory registry
4. Add backend-specific tests in `tests/{backend}/`
5. Update dependencies in `setup.cfg`

### Testing Strategy

- **Unit tests**: Individual operator and component testing
- **Backend tests**: Framework-specific validation
- **Integration tests**: End-to-end model generation and execution
- **CI/CD**: GitHub Actions with pytest across multiple environments

### Constraint System

- Uses Z3 SMT solver for mathematical constraints
- Operators define constraints via `requires()` method
- Type inference via `type_transfer()` method
- Support for custom constraint patches via `patch_requires` configuration

## Important Notes

- TensorFlow can be noisy; use `TF_CPP_MIN_LOG_LEVEL=3` environment variable
- Model outputs default to `nnsmith_output/` directory
- Caches are stored in `~/.cache/nnsmith` for dtype test results
- Some backends (PyTorch/TensorFlow) have conflicting dependencies and must be tested separately
- Graphviz installation required for visualization (`debug.viz=true`)