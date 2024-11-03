
import sys 
import numpy as np
import torch
import meshio
sys.path.append("../..")

from tensormesh import ElementAssembler, NodeAssembler, FacetAssembler, Mesh
from tensormesh import dot, mul
import skfem

class ProductAssembler(FacetAssembler):
    def forward(self, u, v):
        return u * v
    
class LaplaceAssembler(FacetAssembler):
    def forward(self, gradu, gradv):
        return gradu @ gradv 
    

def test_facet_shape():
    mesh = Mesh.gen_rectangle(0.1)
    assembler = ProductAssembler.from_mesh(mesh, quadrature_order=2)
    V = assembler()

    
