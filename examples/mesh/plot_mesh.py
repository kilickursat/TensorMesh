import os 
import sys 
sys.path.append("../..")
import numpy as np
import torch
import matplotlib.pyplot as plt
from tensormesh import mesh,MeshGen
import tensormesh as tm
import tensormesh.visualization as V


def plot_rectangle_mesh():
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(figsize=(12, 6), ncols=2)
    for ele, ax in zip(["tri","quad"], axes):
        mesh_gen = tm.MeshGen(element_type=None, chara_length=0.5, order=4)
        mesh_gen.add_rectangle(0, 0, 1, 1, element=ele)
        mesh_gen.gen().plot(ax=ax)
    axes[0].set_title("Triangle Mesh")
    axes[1].set_title("Quadrilateral Mesh")
    fig.savefig("output/rectangle_mesh.png")

def plot_cube_mesh():
    os.makedirs("output", exist_ok=True)
    fig, axes = plt.subplots(figsize=(12, 6), ncols=2)
    for i, (ele, order, ax) in enumerate(zip(["tet", "hex"], [3,2], axes)):
        mesh_gen = tm.MeshGen(element_type=None, dimension=3, chara_length=0.9, order=order)
        mesh_gen.add_cube(0, 0, 0, 1, 1, 1, element=ele)
        ax = mesh_gen.gen().plot(ax=ax)
        axes[i] = ax
    axes[0].set_title("Tetrahedral Mesh")
    axes[1].set_title("Hexahedral Mesh")
    fig.savefig("output/cube_mesh.png")

def plot_circle_mesh():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=2)
    mesh_gen.add_circle(0.5, 0.5, 0.4, element="quad")
    mesh_gen.gen().plot(save_path="output/circle_mesh.png")

def plot_sphere_mesh():
    os.makedirs("output", exist_ok=True) 
    mesh_gen = tm.MeshGen(element_type=None, dimension=3, chara_length=0.1, order=2)
    mesh_gen.add_sphere(0.5, 0.5, 0.5, 0.4, element="hex")
    mesh_gen.gen().plot(save_path="output/sphere_mesh.png")

def plot_mix_mesh_2d():
    os.makedirs("output", exist_ok=True)
    mesh_gen:tm.Mesh = tm.MeshGen(element_type=None, chara_length=0.1, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh_gen.gen().plot(save_path="output/hybrid_mesh2d.png")

def plot_mix_mesh_3d():
    os.makedirs("output", exist_ok=True)
    mesh_gen:tm.Mesh = tm.MeshGen(element_type=None, chara_length=0.1, order=2)
    mesh_gen.add_cube(0,0,0,0.5,1,1, element="tetra")
    mesh_gen.add_cube(0.5,0,0.5,1,1, element="hex")
    mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh_gen.gen().plot(save_path="output/hybrid_mesh3d.png")

def plot_node_adj_2d():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.node_adjacency())
    fig = ax.get_figure()
    fig.savefig("output/node_adj_2d.png")

def plot_ele_adj_2d():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri") 
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.element_adjacency())
    fig = ax.get_figure()
    fig.savefig("output/ele_adj_2d.png")

def plot_node_adj_3d():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_cube(0,0,0,0.5,1,1, element="tetra")
    mesh_gen.add_cube(0.5,0,0,0.5,1,1, element="hex") 
    mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.node_adjacency())
    fig = ax.get_figure()
    fig.savefig("output/node_adj_3d.png")

def plot_ele_adj_3d():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_cube(0,0,0,0.5,1,1, element="tetra")
    mesh_gen.add_cube(0.5,0,0,0.5,1,1, element="hex")
    mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    ax = V.draw_graph(mesh.element_adjacency())
    fig = ax.get_figure()
    fig.savefig("output/ele_adj_3d.png")


def plot_point_value_2d():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_rectangle(0,0,0.5,1, element="tri")
    mesh_gen.add_rectangle(0.5,0,0.5,1, element="quad")
    mesh_gen.remove_circle(0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    
    # Create some sample point values
    points = mesh.points
    point_values = torch.sin(2*np.pi*points[:,0]) * torch.cos(2*np.pi*points[:,1])
    
    # Plot using interpolation
    img, ax = V.draw_point_value(mesh, point_values)
    fig = ax.get_figure()
    fig.colorbar(img)
    fig.savefig("output/point_value_2d.png")

def plot_point_value_3d():
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.3, order=2)
    mesh_gen.add_cube(0,0,0,0.5,1,1, element="tetra")
    mesh_gen.add_cube(0.5,0,0,0.5,1,1, element="hex")
    mesh_gen.remove_sphere(0.5,0.5,0.5,0.1)
    mesh:tm.Mesh = mesh_gen.gen()
    
    # Create some sample point values
    points = mesh.points
    point_values = torch.sin(2*np.pi*points[:,0]) * torch.cos(2*np.pi*points[:,1]) * torch.sin(2*np.pi*points[:,2])
    
    # Plot using interpolation
    img, ax = V.draw_point_value(mesh, point_values)
    fig = ax.get_figure()
    fig.colorbar(img)
    fig.savefig("output/point_value_3d.png")

def plot_ele_value_2d():
    pass 

def plot_ele_value_3d():
    pass


if __name__ == "__main__":
    # plot_rectangle_mesh()
    plot_cube_mesh()
    exit()
    plot_circle_mesh()
    plot_sphere_mesh()
    plot_mix_mesh_2d()
    plot_mix_mesh_3d()
    plot_node_adj_2d()
    plot_ele_adj_2d()
    plot_node_adj_3d()
    plot_ele_adj_3d()
    plot_point_value_2d()
    plot_point_value_3d()