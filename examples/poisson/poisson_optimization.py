import sys 
sys.path.append("../..")

import torch 
import numpy as np
from tqdm import tqdm
from torch_fem import LaplaceElementAssembler, Mesh,  Condenser, NodeAssembler
from torch_fem.dataset import PoissonMultiFrequency
from torch_fem.visualization import StreamPlotter
import matplotlib.pyplot as plt
from skfem import MeshTri
import meshio

if __name__ == "__main__":
    torch.random.manual_seed(1236)
    # mesh      = Mesh.gen_rectangle(chara_length=0.4)
    
    mesh = MeshTri().refined(2)
    points = mesh.p.T  # Transpose to get an array of points
    cells = {"triangle": mesh.t.T}  # Transpose to get an array of elements
   
    # Create a meshio Mesh object
    boundary_mask = np.zeros(points.shape[0], dtype=bool)
    boundary_mask[mesh.boundary_nodes()] = True
    mesh = meshio.Mesh(points, cells)
    mesh = Mesh.from_meshio(mesh)
    mesh.register_point_data(
        "boundary_mask",
        torch.from_numpy(boundary_mask)
    )
    
    
    K_asm     = LaplaceElementAssembler.from_mesh(mesh)
    equation  = PoissonMultiFrequency(K=8)
    condenser = Condenser(mesh.boundary_mask)

    points    = mesh.points.requires_grad_(True)
    dirchlet_value = torch.zeros(mesh.boundary_mask.shape)
    condenser = Condenser(mesh.boundary_mask, dirchlet_value)

    optimizer = torch.optim.Adam([points], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    epoch = 100

    f = equation.source_term(mesh.points)
    # u = equation.solution(mesh.points)
    class FAssembler(NodeAssembler):
        def forward(self, u, f):
            return u*f
    F_asm = FAssembler.from_mesh(mesh)
    
    loss_fn = torch.nn.MSELoss()

    losses = []

    with StreamPlotter(filename="poisson.mp4") as plotter:
        plotter.draw_mesh_2d(
            points.clone().detach().cpu().numpy(),
            {mesh.default_eletyp:mesh.elements()},
            f)
        pbar = tqdm(total=epoch)
        for i in range(epoch):
            optimizer.zero_grad()
            f_ = F_asm(points, point_data={"f":f})
            K = K_asm(points)
            K, f_ = condenser(K, f_)
            u = K.solve(f_)
            # breakpoint()
            loss = loss_fn(K @ u, f_)
            # TODO: why retain_graph=True?
            # if i == 1:
            #     from torchviz import make_dot 
            #     dot = make_dot(u, params={"points":points,
            #                                 "K":K.edata,
            #                                 "u":u,
            #                                 "f_":f_})
            #     dot.render("computation_graph", format="png")
            #     breakpoint()
            loss.backward(retain_graph=True) 
            
            optimizer.step()
            scheduler.step()
            plotter.draw_mesh_2d(
                points.clone().detach().cpu().numpy(),
                {mesh.default_eletyp:mesh.elements()}, 
                f)
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
            losses.append(loss.item())

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(len(losses)), losses, label="loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig("loss.png")
