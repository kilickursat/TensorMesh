import sys 
sys.path.append("../..")
import tensormesh as tm 
import torch 
import math
from tensormesh.visualization import V
def point_value_2d():
    mesh_gen = tm.MeshGen(element_type=None, dimension=2, chara_length=0.1, order=1)
    mesh_gen.add_cube(0, 0, 0, 1, 1, 1, element="tri")
    ax = mesh_gen.gen().plot(ax=ax, draw_basis=False)

    t = torch.linspace(0, 1, 100)

    for time in t:
        point_values = torch.sin(2 * math.pi * mesh_gen.points[:, 0] * time) * torch.cos(2 * np.pi * mesh_gen.points[:, 1] * time)
        point_values_3d = point_values.unsqueeze(2).repeat(1, 1, 3)
        ax.collections.clear()
        ax = mesh_gen.plot(ax=ax, draw_basis=False, point_values=point_values_3d)
      
    

def point_value_3d():
    pass 

def ele_value_2d():
    pass 

def ele_value_3d():
    pass