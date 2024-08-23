import sys
sys.path.append("../..")

import torch
import numpy as np
from tqdm import tqdm
from torch_fem import LaplaceElementAssembler, Mesh, Condenser
from torch_fem.dataset import PoissonMultiFrequency
from torch_fem.visualization import StreamPlotter
from torch_fem import NodeAssembler
import matplotlib.pyplot as plt
import time
if __name__ == "__main__":
    torch.random.manual_seed(5)
    start_time = time.time()
    mesh      = Mesh.gen_circle(chara_length=0.015)
    assembler = LaplaceElementAssembler.from_mesh(mesh)
    dirchlet_value = torch.zeros(mesh.boundary_mask.shape)
    condenser = Condenser(mesh.boundary_mask, dirchlet_value)
     
    batch_size = 1000
    K = 16
    a = torch.rand((batch_size, K,K)) * 2 - 1
    equation  = PoissonMultiFrequency(a=a)
    mesh.points = mesh.points + 0.5
    fs = equation.source_term(mesh.points,domain="circle") # [batch_size, node]

    K = assembler(mesh.points)

    class FAssembler(NodeAssembler):
        def forward(self, u, fs):
            return u[:, None]*fs[None, :]
    F_asm = FAssembler.from_mesh(mesh)
    fs = F_asm(mesh.points, point_data={"fs":fs.T}) # [batch_size, node]->[node, batch_size]
    fs = fs.reshape(mesh.points.shape[0], -1)  # [node, batch_size]
    
    K, fs = condenser(K, fs)
    
   # u_analytical = equation.solution(mesh.points)
    u = K.solve(fs)
    u = condenser.recover(u)
    fs = condenser.recover(fs)
   
    mesh.plot(
        {"u_fem":u[:,batch_size//2],
         "f":fs[:,batch_size//2],},
        backend="matplotlib",
        save_path="poisson_dataset.png",
        show_mesh=False,
    )
    # save x,y,f,u as a numpy file
    np.savez("poisson_dataset_4000_5000.npz", x=mesh.points[:,0], y=mesh.points[:,1], f=fs, u=u)
    print(time.time()-start_time)
    
    # for i in range(batch_size):
    #     mesh.register_point_data(f"u_fem_{i}", u[:,i])
    #     mesh.register_point_data(f"f_{i}", fs[:,i])
    #     mesh.register_point_data(f"u_analytical_{i}", u_analytical[i])
    # mesh.save("poisson_batch.vtk")


