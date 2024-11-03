import sys 
sys.path.append("../..")

import torch
from tensormesh import ElementAssembler, Mesh,Condenser
from tensormesh.dataset import WaveMultiFrequency

class AAssembler(ElementAssembler):
    def forward(self, gradu, gradv):
        """
            Parameters:
            -----------
                gradu: torch.Tensor[n_dim]
                gradv: torch.Tensor[n_dim]
            Returns:
            --------
                M: torch.Tensor[]
        """
        return gradu @ gradv
    
class MAssembler(ElementAssembler):
    def forward(self, u, v):
        """
            Parameters:
            -----------
                u: torch.Tensor[]
                v: torch.Tensor[]
            Returns:
            --------
                M: torch.Tensor[n_basis]
        """
        return u * v

if __name__ == '__main__':

    dt = 0.001 
    c  = 2.0
    n  = 100
    torch.random.manual_seed(123456)
    
    # mesh = Mesh.gen_rectangle(chara_length=0.01)
    mesh = Mesh.gen_circle(chara_length=0.1, cx=0.5, cy=0.5, r=0.5)
    dataset = WaveMultiFrequency(K=4, c=c)

    u0 = dataset.initial_condition(mesh.points)
    
    M_asm = MAssembler.from_mesh(mesh, quadrature_order=2)
    A_asm = AAssembler.from_mesh(mesh, quadrature_order=2)
    
    M = M_asm()
    A = A_asm()
    condenser = Condenser(mesh.boundary_mask)

    Us  = [u0] 
    v0 = torch.zeros_like(u0) 
    A = c*c*A
    K = 2 * M
    F = -dt * dt * A @ u0 + 2 * M @ u0 + 2 * dt * M @ v0
    K_, F_ = condenser(K, F)
    U_     = K_.solve(F_)
    U      = condenser.recover(U_)
    M_     = condenser(M)[0]
    Us.append(U)
    for _ in range(n-2):
        U1, U2 = Us[-2:]

        F = 2 * M @ U2 - M @ U1 - dt * dt * A @ U2
        
        F_ = condenser.condense_rhs(F)

        U_ = M_.solve(F_)

        U  = condenser.recover(U_)

        Us.append(U)

    Us_gt = [dataset.solution(mesh.points, dt*i) for i in range(n)]
    
    mesh.plot({
        "prediction":Us, 
        "ground truth":Us_gt},
        save_path="wave.mp4", 
        backend="matplotlib", 
        dt=dt, 
        show_mesh=False, 
        linewidth=0.1, 
        linecolor='black')
    
