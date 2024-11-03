import torch 
from typing import Callable, Union
from .elasticity import strain, isotropic_stress, deviatoric_stress, deviatoric_stress_norm
from .ops import divide

def update_plastic_stress(gradu:torch.Tensor, 
                          strain:torch.Tensor, 
                          stress:torch.Tensor,
                          E:Union[float,torch.Tensor] = 70.0,
                          yield_stress:Union[float,torch.Tensor] = 250.0,
                          strain_fn:Callable[[torch.Tensor],torch.Tensor] = strain,
                          stress_fn:Callable[[torch.Tensor,Union[torch.Tensor,float]],torch.Tensor] = isotropic_stress,
                         )->torch.Tensor:

    r"""
    Update stress tensor using plastic constitutive model.

    The plastic model follows von Mises yield criterion with perfect plasticity:

    .. math::

        \sigma_{\text{trial}} = \sigma + \mathbb{C}:\Delta\varepsilon

        f(\sigma_{\text{trial}}) = \|\text{dev}(\sigma_{\text{trial}})\| - \sigma_y

        \Delta\gamma = \frac{\langle f(\sigma_{\text{trial}}) \rangle}{\|\text{dev}(\sigma_{\text{trial}})\|}

        \sigma = \sigma_{\text{trial}} - \Delta\gamma\, \text{dev}(\sigma_{\text{trial}})

    where:
    
    * :math:`\sigma` is the stress tensor in :math:`\mathbb{R}^{D \times D}`
    * :math:`\mathbb{C}` is the elasticity tensor in :math:`\mathbb{R}^{D \times D \times D \times D}`
    * :math:`\varepsilon` is the strain tensor in :math:`\mathbb{R}^{D \times D}`
    * :math:`\sigma_y` is the yield stress scalar in :math:`\mathbb{R}`
    * :math:`\text{dev}` denotes the deviatoric part operator :math:`\mathbb{R}^{D \times D} \rightarrow \mathbb{R}^{D \times D}`
    * :math:`\|\cdot\|` is the von Mises norm operator :math:`\mathbb{R}^{D \times D} \rightarrow \mathbb{R}`
    * :math:`\langle \cdot \rangle` denotes the positive part operator :math:`\mathbb{R} \rightarrow \mathbb{R}`

    The model uses a trial elastic predictor followed by plastic correction if yielding occurs.
    If the trial stress exceeds the yield surface, it is projected back onto the yield surface.
    
    Parameters
    ----------
    gradu : torch.Tensor
        1D Tensor of shape [d], where d is the spatial dimension.
        Gradient of displacement field with respect to spatial coordinates.
    strain : torch.Tensor 
        2D Tensor of shape [d, d], where d is the spatial dimension.
        Current strain tensor at the start of the timestep.
    stress : torch.Tensor
        2D Tensor of shape [d, d], where d is the spatial dimension.
        Current stress tensor at the start of the timestep.
    E : Union[float, torch.Tensor], default=70.0
        Young's modulus. If tensor, must be 0D scalar tensor.
        Controls the elastic stiffness of the material.
    yield_stress : Union[float, torch.Tensor], default=250.0
        Yield stress threshold. If tensor, must be 0D scalar tensor.
        Material yields plastically when von Mises stress exceeds this value.
    strain_fn : Callable[[torch.Tensor], torch.Tensor], default=strain
        Function to compute strain tensor from displacement gradient.
        Default uses small strain assumption:
        
        .. math::
        
            \varepsilon_{ij} = \frac{1}{2}(\nabla u_{ij} + \nabla u_{ji}), \quad \varepsilon,\nabla u \in \mathbb{R}^{d \times d}
            
    stress_fn : Callable[[torch.Tensor, Union[float,torch.Tensor]], torch.Tensor], default=isotropic_stress
        Function to compute stress tensor from strain tensor and Young's modulus.
        Default uses isotropic linear elasticity:
        
        .. math::
        
            \sigma_{ij} = \lambda \text{tr}(\varepsilon)\delta_{ij} + 2\mu\varepsilon_{ij}, \quad \sigma,\varepsilon \in \mathbb{R}^{d \times d}
            
        where :math:`\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}`, :math:`\mu = \frac{E}{2(1+\nu)}`, :math:`E,\nu \in \mathbb{R}`, and :math:`\delta_{ij}` is the Kronecker delta

    Returns
    -------
    torch.Tensor
        2D Tensor of shape [d, d], where d is the spatial dimension.
        Updated stress tensor after plastic correction.
    """
    # assertion
    if isinstance(E, torch.Tensor):
        assert E.numel() == 1
    if isinstance(yield_stress, torch.Tensor):
        assert yield_stress.numel() == 1
    assert gradu.dim() == 1, f"gradu should be a 1D tensor of shape [dim], but got shape {gradu.shape}"
    assert strain.dim() == 2, f"strain should be a 2D tensor of shape [dim, dim], but got shape {strain.shape}"
    assert stress.dim() == 2, f"stress should be a 2D tensor of shape [dim, dim], but got shape {stress.shape}"
    assert strain.shape == stress.shape, f"strain and stress should have same shape, but got {strain.shape} and {stress.shape}"
    assert strain.shape[0] == strain.shape[1], f"strain should be square matrix, but got shape {strain.shape}"
    assert gradu.shape[0] == strain.shape[0], f"gradu dimension should match strain dimension, but got {gradu.shape[0]} and {strain.shape[0]}"
    # get stress trial
    delta_strain = strain_fn(gradu) - strain # [dim, dim]
    assert delta_strain.shape == strain.shape, f"delta_strain should have same shape as strain, but got {delta_strain.shape} and {strain.shape}"
    stress_trial = stress_fn(delta_strain, E) + stress # [dim, dim]
    assert stress_trial.shape == stress.shape, f"stress_trial should have same shape as stress, but got {stress_trial.shape} and {stress.shape}"
    # yield function
    stress_devia = deviatoric_stress(stress_trial) # [dim, dim]
    stress_devia_norm = deviatoric_stress_norm(stress_trial) # []
    f_yield = stress_devia_norm - yield_stress
    f_yield = torch.clamp_min(f_yield, 0.)

    # update stress
    stress = stress_trial - f_yield * divide(stress_devia , stress_devia_norm)

    return stress
    
