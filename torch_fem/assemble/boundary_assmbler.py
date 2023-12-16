
from .node_assembler import NodeAssembler


class BoundaryAssmebler(NodeAssembler):

    @classmethod
    def from_mesh(cls,  mesh, boundary_mask=None, quadrature_order=None):
        r"""
        Parameters
        ----------
            mesh: Mesh
            boundary_mask: torch.Tensor[n_points,]
                If None, use the property of mesh.boundary_mask
            quadrature_order: int
                default is None
        """
        dim = max(mesh.dim2eletyp.keys())
        boundary_dim = dim - 1
        element_types = mesh.dim2eletyp[boundary_dim]
        boundary_elements = mesh.elements(element_types)

        boundary_mask = mesh.boundary_mask if boundary_mask is None else boundary_mask
        for element_type in element_types:
            boundary_element = boundary_elements[element_type] # [n_element, n_vertex_per_element]
            boundary_element = boundary_element[boundary_element.all(-1)]
            if boundary_elements.numel() > 0:
                boundary_elements[element_type] = boundary_element
            else:
                boundary_elements.pop(element_type)

        assert len(boundary_elements) > 0, "No boundary element found, make sure the mask is correct"


        n_points = mesh.n_points
        return cls.from_elements(boundary_elements, n_points, quadrature_order, device=mesh.device, dtype=mesh.dtype)