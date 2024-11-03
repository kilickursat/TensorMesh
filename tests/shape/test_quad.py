import pytest
import torch
import sys 

sys.path.append("../..")
from tensormesh.element import Quadrilateral


def test_sum_p1():
    x, y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = Quadrilateral.get_basis_fns(1)(xy) # [n_quadrature, n_basis]
    phi  = phi.sum(dim=-1)
    assert torch.allclose(phi, torch.ones_like(phi))

def test_max_p1():
    x, y = torch.meshgrid(torch.linspace(0, 1, 200), torch.linspace(0, 1, 200))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = Quadrilateral.get_basis_fns(1)(xy) # [n_quadrature, n_basis]
    phi  = phi.max(dim=0).values # [n_basis]
    assert torch.allclose(phi, torch.ones_like(phi))

def test_sum_p2():
    x, y = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = Quadrilateral.get_basis_fns(2)(xy) # [n_quadrature, n_basis]
    phi  = phi.sum(dim=-1)
    assert torch.allclose(phi, torch.ones_like(phi))

def test_max_p2():
    x, y = torch.meshgrid(torch.linspace(0, 1, 400), torch.linspace(0, 1, 400))
    xy   = torch.stack([x, y], dim=-1).reshape(-1, 2)
    phi  = Quadrilateral.get_basis_fns(2)(xy) # [n_quadrature, n_basis]
    phi  = phi.max(dim=0).values # [n_basis]
    assert torch.allclose(phi, torch.ones_like(phi),  rtol=1e-4), f"phi = {phi}"

def test_integral_p1():
    # TODO: ? integral is -0.5
    # Create a fine grid of points for numerical integration
    x, y = torch.meshgrid(torch.linspace(0, 1, 400), torch.linspace(0, 1, 400))
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    # Get quadrature points and gradients of basis functions
    quadrature = torch.stack([x, y], dim=-1)  # [n_points, 2]
    grad = Quadrilateral.get_basis_grad_fns(1).map(quadrature)  # [n_points, n_basis, 2]
    
    # For each basis function, verify that integral of gradient is zero
    # This is expected since basis functions are zero on the boundary
    n_basis = grad.shape[1]
    for i_basis in range(n_basis):
        # Integrate gradient components using trapezoidal rule
        integral_dx = torch.trapz(grad[:, 0, i_basis], x, dim=0)  # Integral of d/dx
        integral_dy = torch.trapz(grad[:, 1, i_basis], y, dim=0)  # Integral of d/dy
        
        # Check integrals are zero (up to numerical precision)
        breakpoint()
        assert torch.allclose(integral_dx, torch.zeros_like(integral_dx), atol=1e-3)
        assert torch.allclose(integral_dy, torch.zeros_like(integral_dy), atol=1e-3)

def test_integral_p2():
    x, y = torch.meshgrid(torch.linspace(0, 1, 400), torch.linspace(0, 1, 400))
    x  = x.reshape(-1)
    y  = y.reshape(-1)
   
    element_coords = basis_p2

    quadrature = torch.stack([x, y], dim=-1)
    grad = shape_grad_p2(torch.stack([x, y], dim=-1), element_coords[None, :, :], return_jac=False)[0] # [n_quadrature, n_basis, n_dim]

    n_basis = grad.shape[1]
    for i_basis in range(n_basis):

        integral_dx = torch.trapz(grad[:, i_basis, 0], quadrature[:, 0], dim=0) # []
        integral_dy = torch.trapz(grad[:, i_basis, 1], quadrature[:, 1], dim=0) # []

        torch.allclose(integral_dx, torch.zeros_like(integral_dx))
        torch.allclose(integral_dy, torch.zeros_like(integral_dy))