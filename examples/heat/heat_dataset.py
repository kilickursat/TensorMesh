import sys 
sys.path.append("../..")

import torch
import numpy as np
from tensormesh import ElementAssembler, Mesh,Condenser
from tensormesh.dataset import HeatMultiFrequency

import time
class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_basis, n_dim]
                gradv: torch.Tensor[n_basis, n_dim]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
       
        return gradu @ gradv.T
    
class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[n_basis]
                v: torch.Tensor[n_basis]
            Returns:
            --------
                M: torch.Tensor[n_basis, n_basis]
        """
        return torch.outer(u, v)

if __name__ == '__main__':
    torch.random.manual_seed(3)
    file_name = "heat_dataset_2000_3000.npz"
    #mesh = Mesh.gen_rectangle(chara_length=0.02,order=2, element_type="tri")
    mesh = Mesh.gen_L(chara_length=0.008, element_type="tri")
    time_start = time.time() 
    batch_size = 1000
    d = 16
    mus = torch.rand((batch_size, d))
    dataset = HeatMultiFrequency(mu=mus)
    u0s = dataset.initial_condition(mesh.points)
    
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    
    # M = M_asm(mesh.points)
    # A = A_asm(mesh.points)
    M = M_asm() 
    A = A_asm()
    
    dirchlet_value = torch.zeros(mesh.boundary_mask.shape)
    condenser = Condenser(mesh.boundary_mask, dirchlet_value)

    U = u0s 
    dt = 0.00005
    D  = 1
    n  = 100
    K  = M + dt * D * D * A 
    K_ = condenser(K)[0]
    U = U.T
    Us = [U]
    for _ in range(n-1):
        F = M @ U # [num_node, num_batch]

        F_ = condenser.condense_rhs(F)
        
        U_ = K_.solve(F_)

        U  = condenser.recover(U_)

        Us.append(U)
    
    #Us_gts = [dataset.solution(mesh.points, dt*i) for i in range(n)]
    

    Us = torch.stack(Us, dim=1).permute(2,1,0)
    #Us_gts = torch.stack(Us_gts, dim=1)

    idx = 8
    U_list = [Us[idx][i,:] for i in range(100)]
    #Us_gts_list = [Us_gts[idx][i,:] for i in range(100)]
    mesh.plot(
        {"prediction":U_list},
        save_path="heat_dataset.mp4", 
        backend="matplotlib", 
        dt=dt,
        show_mesh=False)
    print("Time cost: ", time.time()-time_start)
    np.savez(file_name, x=mesh.points[:,0], y=mesh.points[:,1], u0 = u0s, u=Us, dt = dt)
