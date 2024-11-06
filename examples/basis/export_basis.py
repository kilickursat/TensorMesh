import sys
sys.path.append("../..")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.tools as tls
import plotly.graph_objects as go
from scipy.interpolate import griddata
from tensormesh.element import Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism
from tensormesh import mesh,MeshGen
import tensormesh as tm
import tensormesh.visualization as V
import tensormesh.element as E
from tensormesh.element.plot import plot_1d, plot_2d, plot_3d


elements = [Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism]
    
def plot_basis_1d():

    fig, axes = plt.subplots(ncols=4, figsize=(16,4))

    for i, order in enumerate(range(1,5)):
        basis = Line.get_basis(order)
        basis_fns = Line.get_basis_fns(order)
        plot_1d(basis, basis_fns, ax=axes[i], legend=False)
        axes[i].set_title(f"Order:{order}")

    plt.title("Line Basis Functions")
    plt.tight_layout()
    os.makedirs("../../../tensormesh-docs/source/_static/plot_basis", exist_ok=True)
    plt.savefig("../../../tensormesh-docs/source/_static/plot_basis/linear.png")

def plot_basis_2d():

    font_size = 14
    scatter_size = 100  # Control size of scatter points

    for name, element in zip(["triangle","quadrilateral"],[Triangle,Quadrilateral]):
        fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
        for order in range(1, 5):
            ax = axes[order - 1]
            basis = element.get_basis(order)
            n_basis = basis.shape[0]
            for i in range(n_basis):
                ax.scatter(basis[i, 0], basis[i, 1], s=scatter_size)
                ax.text(basis[i, 0], basis[i, 1], f'{i+1}', fontsize=font_size)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)
        
            # Draw element edges
            edges = element.points[element.edge]
            for edge in edges:
                ax.plot(edge[:, 0], edge[:, 1], 'k-', alpha=0.5)
        plt.tight_layout()
        os.makedirs("../../../tensormesh-docs/source/_static/plot_basis", exist_ok=True)
        plt.savefig(f"../../../tensormesh-docs/source/_static/plot_basis/{name}.png")
        plt.close()

def plot_basis_3d():
    # Generate basis data

    def draw_basis_3d_by_element(element, name):
        basis = element.get_basis(order=3)


        node_trace = go.Scatter3d(
            x=basis[:,0].flatten(),
            y=basis[:,1].flatten(),
            z=basis[:,2].flatten(),
            mode='markers+text',
            marker=dict(size=10, 
                        color=np.arange(basis.shape[0]), 
                        colorscale='Viridis', 
                        opacity=0.8),
            text=[f'{i+1}' for i in range(basis.shape[0])],
            textposition='top center'
        )

        edge_trace = go.Scatter3d(
            x=element.points[element.edge][..., 0].flatten(),
            y=element.points[element.edge][..., 1].flatten(),
            z=element.points[element.edge][..., 2].flatten(),
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )

        # Create plot
        fig = go.Figure(data=[node_trace, edge_trace])
        
        fig.update_layout(
            title=f'3D Basis of {name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Basis Value'
            )
        )
    
        # Save as HTML
        os.makedirs("../../../tensormesh-docs/source/source/_static/plot_basis", exist_ok=True)
        fig.write_html(f"../../../tensormesh-docs/source/source/_static/plot_basis/{name}.html")

    for name, element in [
        ("tetra", Tetrahedron),
        ("hex", Hexahedron),
        ("pyr", Pyramid),
        ("pri", Prism)
        ]:
        draw_basis_3d_by_element(element, name)

if __name__ == '__main__':
    plot_basis_1d()
    plot_basis_2d()
    plot_basis_3d()