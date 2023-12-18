from .matrix import SparseMatrix

from .utils import is_petsc_available, is_cupy_available
from .solve import is_cpp_backend_available as is_solve_cpp_backend_available