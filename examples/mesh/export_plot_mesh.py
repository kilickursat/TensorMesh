import os
import torch 
import numpy as np
import plotly.io as pio
import plotly.tools as tls
import plotly.graph_objects as go
from scipy.interpolate import griddata
from plot_mesh import (
    plot_rectangle_mesh,
    plot_cube_mesh, 
    plot_circle_mesh,
    plot_mix_mesh_2d,
    plot_node_adj_2d,
    plot_ele_adj_2d,
    plot_node_adj_3d,
    plot_ele_adj_3d,
    plot_point_value_2d,
    plot_point_value_3d,
    plot_ele_value_2d,
    plot_ele_value_3d,
    GRAPH_COLOR,
    GRAPH_LINEWIDTH,
    GRAPH_ALPHA,
    MESH_COLOR,
    MESH_LINEWIDTH,
    tm, E 
)

# Create output directory
os.makedirs("../../docs/source/_static/plot_mesh", exist_ok=True)

# Generate and export each plot
plots = {
    'rectangle_mesh': plot_rectangle_mesh,
    'cube_mesh': plot_cube_mesh,
    'circle_mesh': plot_circle_mesh,
    'mix_mesh_2d': plot_mix_mesh_2d,
    'node_adj_2d': plot_node_adj_2d,
    'ele_adj_2d': plot_ele_adj_2d,
    'node_adj_3d': plot_node_adj_3d,
    'ele_adj_3d': plot_ele_adj_3d,
    'point_value_2d': plot_point_value_2d,
    'point_value_3d': plot_point_value_3d,
    'ele_value_2d': plot_ele_value_2d,
    'ele_value_3d': plot_ele_value_3d
}


def plot_cube_mesh(return_fig=False):
    # Create mesh
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    mesh = mesh_gen.gen()
    
    # Get mesh edges
    elements = mesh.elements()
    edges = set()
    for tet in elements:
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([tet[i], tet[j]]))
                edges.add(edge)
    
    # Create edge traces
    points = mesh.points
    x_edges = []
    y_edges = []
    z_edges = []

    # Create node trace
    node_trace = go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(
            size=6,
            color=MESH_COLOR,
            opacity=0.8
        ),
        name='Nodes'
    )
    
    for i, j in edges:
        x_edges.extend([points[i,0], points[j,0], None])
        y_edges.extend([points[i,1], points[j,1], None])
        z_edges.extend([points[i,2], points[j,2], None])
    
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color=MESH_COLOR, width=MESH_LINEWIDTH),
        name='Mesh'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace,node_trace])
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        )
    )
    

    if return_fig:
        return fig
    else:
        fig.write_html("../../docs/source/_static/plot_mesh/cube_mesh.html")

def plot_node_adj_3d(return_fig=False):

    
    os.makedirs("output", exist_ok=True)
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    mesh = mesh_gen.gen()
    
    # Get node adjacency and points
    adj = mesh.node_adjacency()
    points = mesh.points
    
    # Create edges for graph visualization
    edges_i, edges_j = adj.col, adj.row
    x_edges = []
    y_edges = []
    z_edges = []
    
    for i, j in zip(edges_i, edges_j):
        x_edges.extend([points[i,0], points[j,0], None])
        y_edges.extend([points[i,1], points[j,1], None])
        z_edges.extend([points[i,2], points[j,2], None])
    
    # Create graph edges trace
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color=GRAPH_COLOR, width=5),
        opacity=0.3,
        name='Node Adjacency'
    )
    
    # Create nodes trace
    node_trace = go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(
            size=6,
            color='orange',
        ),
        opacity = 0.3,
        name='Nodes'
    )
    
    # Create mesh edges
    elements = mesh.elements()
    edges = set()
    for tet in elements:
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([int(tet[i]), int(tet[j])]))
                edges.add(edge)
                
    x_mesh = []
    y_mesh = []
    z_mesh = [] 
    for i,j in edges:
        x_mesh.extend([points[i,0], points[j,0], None])
        y_mesh.extend([points[i,1], points[j,1], None])
        z_mesh.extend([points[i,2], points[j,2], None])
        
    # Create mesh edges trace
    mesh_trace = go.Scatter3d(
        x=x_mesh, y=y_mesh, z=z_mesh,
        mode='lines',
        line=dict(color=MESH_COLOR, width=MESH_LINEWIDTH),
        opacity=0.2,
        name='Mesh'
    )
    
    # Create figure
    fig = go.Figure(data=[node_trace, mesh_trace,edge_trace])
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z'
        )
    )
    
    if return_fig:
        return fig
    
    fig.show()
    fig.write_html("output/node_adj_3d.html")

def plot_ele_adj_3d(return_fig=False):
    # Get mesh data
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    mesh = mesh_gen.gen()

    points = mesh.points
    elements = mesh.elements()

    # Get node adjacency and points
    adj = mesh.element_adjacency()
    points = mesh.points
    
    # Create edges for graph visualization
    edges_i, edges_j = adj.col, adj.row
    x_edges = []
    y_edges = []
    z_edges = []
    
    points = mesh.points[mesh.elements()].mean(1)
    for i, j in zip(edges_i, edges_j):
        x_edges.extend([points[i,0], points[j,0], None])
        y_edges.extend([points[i,1], points[j,1], None])
        z_edges.extend([points[i,2], points[j,2], None])
    
    # Create graph edges trace
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color=GRAPH_COLOR, width=6),
        opacity=1.0,
        name='Element Adjacency'
    )
    
    # Create nodes trace
    node_trace = go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(
            size=6,
            color='orange',
            opacity = 0.3
        ),
        name='Nodes'
    )
    

    # Create mesh edges
    edges = E.Tetrahedron.element_to_edge(elements, order=1)
    points = mesh.points
    x_mesh = []
    y_mesh = []
    z_mesh = [] 
    for i,j in edges:
        x_mesh.extend([points[i,0], points[j,0], None])
        y_mesh.extend([points[i,1], points[j,1], None])
        z_mesh.extend([points[i,2], points[j,2], None])
        
    # Create mesh edges trace
    mesh_trace = go.Scatter3d(
        x=x_mesh, y=y_mesh, z=z_mesh,
        mode='lines',
        line=dict(color=MESH_COLOR, width=MESH_LINEWIDTH),
        opacity=0.1,
        name='Mesh'
    )
    
    # Create figure
    fig = go.Figure(data=[node_trace, edge_trace, mesh_trace])
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    if return_fig:
        return fig
        
    fig.show()
    breakpoint()
    fig.write_html("output/ele_adj_3d.html")

def plot_point_value_3d(return_fig=False):
    # Get mesh
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    mesh = mesh_gen.gen()
    
    points = mesh.points.cpu().numpy()
    elements = mesh.elements()
    
    # Create random values for points
    point_values = np.sin(2*np.pi*points[:,0]) * np.cos(2*np.pi*points[:,1]) * np.sin(2*np.pi*points[:,2])

    # Define grid points where interpolation is desired
    grid_x, grid_y, grid_z = np.mgrid[
        points[:, 0].min():points[:, 0].max():20j,
        points[:, 1].min():points[:, 1].max():20j,
        points[:, 2].min():points[:, 2].max():20j
    ]


    use_vol = False 

    if use_vol:
        # Interpolate point values onto the grid
        grid_values = griddata(points, point_values, (grid_x, grid_y, grid_z), method='linear')

        # Create scatter trace for grid data
        grid_trace = go.Volume(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            z=grid_z.flatten(),
            value=grid_values.flatten(),
            opacity=0.1,  # Set opacity for the grid data
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Grid Values'),
            name='Grid Data'
        )
    
        # Create figure
        fig = go.Figure(data=[grid_trace])
    
    else:

        grid_values = griddata(points, point_values, (grid_x, grid_y, grid_z), method='linear')

        # Create scatter trace for point data
        scatter_trace = go.Scatter3d(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            z=grid_z.flatten(),
            mode='markers',
            marker=dict(
                size=7,
                color=grid_values.flatten(),
                colorscale='Viridis',
                opacity=0.3,
                colorbar=dict(title='Grid Values')
            ),
            name='Grid Data'
        )
        # Create figure
        fig = go.Figure(data=[scatter_trace])
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z'
        )
    )
    
    if return_fig:
        return fig
        
    fig.show()
    fig.write_html("output/point_value_3d.html")

def plot_ele_value_3d(return_fig=False,
                      method:str = "linear"):
    # Get mesh
    mesh_gen = tm.MeshGen(element_type=None, chara_length=0.1, order=1, dimension=3)
    mesh_gen.add_cube(0,0,0,1,1,1,element="tet")
    mesh = mesh_gen.gen()
    
    points = mesh.points.cpu().numpy()
    elements = mesh.elements().cpu().numpy()
    
    # Create random values for elements
    # Extract just the tensor values from the dictionary
    element_values = np.sin(2*np.pi*points[elements].mean(1)[:,0]) * \
                    np.cos(2*np.pi*points[elements].mean(1)[:,1])
    
    # Get centroids of elements
    centroids = points[elements].mean(1)

    element_values = np.sin(2 * np.pi * centroids[:, 0]) * \
                     np.cos(2 * np.pi * centroids[:, 1])

    # Define grid for interpolation
    grid_x, grid_y, grid_z = np.mgrid[
        points[:,0].min():points[:,0].max():20j,
        points[:,1].min():points[:,1].max():20j,
        points[:,2].min():points[:,2].max():20j
    ]


    if method == "nearest":
        # Interpolate element values onto the grid
        grid_values = griddata(centroids, element_values, (grid_x, grid_y, grid_z), method='nearest')

        # Create volume trace for grid data
        grid_trace = go.Volume(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            z=grid_z.flatten(),
            value=grid_values.flatten(),
            opacity=0.5,  # Set opacity for the grid data
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Grid Values'),
            name='Grid Data'
        )

        # Create figure
        fig = go.Figure(data=[grid_trace])

    elif method in ["linear", "cubic"]:

        grid_values = griddata(centroids, element_values, (grid_x, grid_y, grid_z), method=method)

        # Filter out NaN values from grid_values
        valid_mask = ~np.isnan(grid_values)
        valid_x = grid_x[valid_mask]
        valid_y = grid_y[valid_mask]
        valid_z = grid_z[valid_mask]
        valid_values = grid_values[valid_mask]

        # Create scatter trace for valid grid data
        scatter_trace = go.Scatter3d(
            x=valid_x.flatten(),
            y=valid_y.flatten(),
            z=valid_z.flatten(),
            mode='markers',
            marker=dict(
                size=7,
                color=valid_values.flatten(),
                colorscale='Viridis',
                opacity=0.2,
                colorbar=dict(title='Grid Values')
            ),
            name='Valid Grid Data'
        )
        # Create figure
        fig = go.Figure(data=[scatter_trace])

    
   
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    if return_fig:
        return fig
        
    fig.show()
    fig.write_html("output/ele_value_3d.html")

def export_by_plotly():

    for name, plot_func in plots.items():
        # Get matplotlib figure
        mpl_fig = plot_func(return_fig=True)
        
        # Convert matplotlib figure to plotly figure
        try:
            plotly_fig = tls.mpl_to_plotly(mpl_fig)
        except:
            breakpoint()
        # Update layout for better display
        plotly_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )
        
        # Save as HTML
        html = plotly_fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={'responsive': True}
        )
        
        with open(f"../../docs/source/_static/plot_mesh/{name}.html", "w") as f:
            f.write(html)

def export_by_mpld3():
    import mpld3
    for name, plot_func in plots.items():
        # Get matplotlib figure
        mpl_fig = plot_func(return_fig=True)
        
        # Convert matplotlib figure to mpld3 figure
        d3_fig = mpld3.fig_to_html(mpl_fig)
        
        # Save as HTML
        with open(f"../../docs/source/_static/plot_mesh/{name}_mpld3.html", "w") as f:
            f.write(d3_fig)

plots = {
    'rectangle_mesh': plot_rectangle_mesh,
    'cube_mesh': plot_cube_mesh,
    'circle_mesh': plot_circle_mesh,
    'mix_mesh_2d': plot_mix_mesh_2d,
    'node_adj_2d': plot_node_adj_2d,
    'ele_adj_2d': plot_ele_adj_2d,
    'node_adj_3d': plot_node_adj_3d,
    'ele_adj_3d': plot_ele_adj_3d,
    'point_value_2d': plot_point_value_2d,
    'point_value_3d': plot_point_value_3d,
    'ele_value_2d': plot_ele_value_2d,
    'ele_value_3d': plot_ele_value_3d
}

def export():
    for name, plot_func in plots.items():
        # Get figure
        fig = plot_func(return_fig=True)
        
        # Check if 3D by looking for Scatter3d traces
        is_3d = isinstance(fig, go.Figure)
        
        if is_3d:
            # For 3D plots, use plotly figure directly
            html = fig.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                config={'responsive': True}
            )

             # Save HTML
            with open(f"../../docs/source/_static/plot_mesh/{name}.html", "w") as f:
                f.write(html)
        else:
            # For 2D plots, save as PNG
            try:
                fig.savefig(f"../../docs/source/_static/plot_mesh/{name}.png", 
                           bbox_inches='tight',
                           dpi=300)
            except:
                print(f"Failed to save {name} as PNG")
                continue
       

if __name__ == '__main__':
    export()