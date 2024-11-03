import torch 
import sys 
sys.path.append("../..")

from tensormesh import ElementAssembler,FacetAssembler, Mesh,Condenser

from tensormesh.functional import sym,eye,trace,ddot

class KAssembler(ElementAssembler):
    def __post_init__(self):
        self.d = 0.1
        self.E = 200e9
        self.nu = 0.3
    def forward(self, u, v, gradu, gradv):
        """
            Parameters:
            -----------
                u: torch.Tensor[]
                v: torch.Tensor[]
                gradu: torch.Tensor[n_dim]
                gradv: torch.Tensor[n_dim]
            Returns:
            --------
                K: torch.Tensor[n_dim, n_dim]
        """
       
        n_dim = gradu.shape[0]
        strain_u = sym(gradu) # [n_dim, n_dim]
        breakpoint()
        stress_u = self.E / (1 + self.nu) * (strain_u + self.nu / (1 - self.nu) * eye(trace(strain_u), n_dim))   # [n_dim, n_dim]
        strain_v = sym(gradv) # [n_dim, n_dim]

        return self.d**3 / 12.0 * (stress_u * strain_v).sum()
    
