:github_url: https://github.com/walkerchi/torch_fem 

Tensormesh Document 
===================


.. code-block:: none

    ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗███████╗██╗  ██╗
    ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔════╝██║  ██║
       ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██╔████╔██║█████╗  ███████╗███████║
       ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║
       ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗███████║██║  ██║
       ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝

TensorMesh: The Modern FEM Library 🚀
-------------------------------------

A fast 🚀, differentiable 🎯, cross-platform 💻, JIT-free 📌, and debugging-friendly 🚨 finite element library that prioritizes user experience through clean, Pythonic APIs 🤗

TensorMesh is a modern finite element method (FEM) library built on PyTorch, designed to solve partial differential equations (PDEs) with elegance and efficiency. By seamlessly integrating with PyTorch's ecosystem, it provides automatic differentiation and GPU acceleration while maintaining an intuitive, Pythonic interface.


Core Strengths
--------------

- **Easy to Use**: Clean, intuitive Pythonic APIs that make FEM accessible to both beginners and experts
- **Easy to Debug**: Clear error messages and straightforward execution flow for painless debugging
- **Cross Platform**: Works seamlessly across Windows, Linux, and macOS without complex dependencies
- **Seamless PyTorch Integration**: Leverage PyTorch's powerful automatic differentiation and GPU acceleration
- **Comprehensive Element Support**: Work with a wide range of elements including triangular, tetrahedral, pyramid, and prismatic types
- **High-Performance Assembly**: Optimized element assembly operations for both CPU and GPU architectures
- **Advanced Solvers**: Efficient sparse matrix solvers with flexible backend options (PETSc, PyTorch)
- **Rich Visualization**: Integrated tools for mesh and solution visualization
- **Smart Mesh Generation**: Automated mesh generation for common geometries with intelligent defaults

Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15

   * - Feature
     - FEniCS
     - scikit-fem
     - JAX-FEM
     - TensorMesh
   * - Flexibility
     - ❌
     - ✅
     - ❌
     - ✅
   * - Easy Install
     - ❌
     - ✅
     - ✅
     - ✅
   * - Easy Debug
     - ❌
     - ✅
     - ❌
     - ✅
   * - Easy IO
     - ❌
     - ❌
     - ❌
     - ✅
   * - Large Mesh
     - ✅
     - ✅
     - ❌
     - ✅
   * - GPU Support
     - ✅
     - ❌
     - ✅
     - ✅
   * - Efficiency
     - ✅
     - ❌
     - ✅
     - ✅
   * - Auto-diff
     - ✅
     - ❌
     - ✅
     - ✅
   * - DL Integration
     - ❌
     - ❌
     - ✅
     - ✅


.. toctree::
   :maxdepth: 1 
   :caption: Get Started

   get_started/installation
   get_started/benchmark
   get_started/element

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/mesh_gen
   examples/adjacency
   examples/basis
   examples/values
   examples/poisson
   examples/wave
   examples/linear_elasticity


.. toctree::
   :maxdepth: 5
   :caption: API Reference

   api_reference/dataset
   api_reference/mesh
   api_reference/element
   api_reference/assemble
   api_reference/sparse
   api_reference/operator
   api_reference/ode
   api_reference/functional
   api_reference/nn 
   api_reference/visualization
   api_reference/profile
   