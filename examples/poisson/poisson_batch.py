import sys
sys.path.append("../..")

import torch
import numpy as np
from tqdm import tqdm
from tensormesh import LaplaceElementAssembler, Mesh, Condenser
from tensormesh.dataset import PoissonMultiFrequency
from tensormesh.visualization import StreamPlotter
from tensormesh import NodeAssembler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.random.manual_seed(1)
    #mesh      = Mesh.gen_circle(chara_length=0.015)
    mesh    = Mesh.gen_rectangle(chara_length=0.1)
    # mesh = Mesh.gen_rectangle(chara_length=0.02, order=2, element_type="tri")
    
    assembler = LaplaceElementAssembler.from_mesh(mesh)
    dirchlet_value = torch.zeros(mesh.boundary_mask.shape)
    condenser = Condenser(mesh.boundary_mask, dirchlet_value)
     
    batch_size = 16
    K = 4
    a = torch.rand((batch_size, K,K)) * 2 - 1
    equation  = PoissonMultiFrequency(a=a)
    
    fs = equation.source_term(mesh.points,domain="rectangle") # [batch_size, node]

    K = assembler(mesh.points)
    class FAssembler(NodeAssembler):
        def forward(self, u, fs):
            return u[:, None]*fs[None, :]
    F_asm = FAssembler.from_mesh(mesh)
    fs = F_asm(mesh.points, point_data={"fs":fs.T}) # [batch_size, node]->[node, batch_size]
    fs = fs.reshape(mesh.points.shape[0], -1)  # [node, batch_size]
    
    K, fs = condenser(K, fs)
    
    u_analytical = equation.solution(mesh.points)
    u = K.solve(fs)
    u = condenser.recover(u)
    fs = condenser.recover(fs)
   
    mesh.plot(
        {"u_fem":u[:,batch_size//2],
         "f":fs[:,batch_size//2],
         "u_analytical":u_analytical[batch_size//2]},
        backend="matplotlib",
        save_path="poisson_batch.png",
        show_mesh=True,
    )
    
    for i in range(batch_size):
        mesh.register_point_data(f"u_fem_{i}", u[:,i])
        mesh.register_point_data(f"f_{i}", fs[:,i])
        mesh.register_point_data(f"u_analytical_{i}", u_analytical[i])
    mesh.save("poisson_batch.vtk")


