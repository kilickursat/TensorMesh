
import sys
import torch
sys.path.append("../..")

from tensormesh import FacetAssembler, Mesh


class ProductAssembler(FacetAssembler):
    def forward(self, u, v):
        return u * v


class LaplaceAssembler(FacetAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv


class IntegrateOne(FacetAssembler):
    def forward(self, v):
        return v


def test_facet_shape():
    mesh = Mesh.gen_rectangle(0.1)
    assembler = ProductAssembler.from_mesh(mesh, quadrature_order=2)
    V = assembler()
    assert V.shape == (mesh.points.shape[0],)


def test_facet_integrate_one_p1_triangle():
    # On a unit-square mesh the boundary integral of 1 must equal the perimeter.
    mesh = Mesh.gen_rectangle(0.1)
    val = IntegrateOne.from_mesh(mesh)().sum().item()
    assert abs(val - 4.0) < 1e-10


def test_facet_integrate_one_p2_triangle():
    mesh = Mesh.gen_rectangle(0.2, order=2)
    val = IntegrateOne.from_mesh(mesh)().sum().item()
    assert abs(val - 4.0) < 1e-10


def test_facet_reduce_sparse_match():
    mesh = Mesh.gen_rectangle(0.1)
    v_reduce = IntegrateOne.from_mesh(mesh, project="reduce")().sum().item()
    v_sparse = IntegrateOne.from_mesh(mesh, project="sparse")().sum().item()
    assert abs(v_reduce - v_sparse) < 1e-10
