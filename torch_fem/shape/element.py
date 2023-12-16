import torch
import re 
from typing import List

class Element:
    points: torch.Tensor
    vertex: torch.Tensor
    edge: torch.Tensor
    face: List[List[int]]
    cell: torch.Tensor
    dim: int
    n_vertex: int
    n_edge: int
    n_face: int
    n_cell: int

class Line(Element):
    points = torch.tensor([[0.0],[1.0]]) # 2x1
    vertex = torch.tensor([[0], [1]]) # 2x1
    edge   = torch.tensor([[0, 1]]) # 1x2
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = 0
    n_cell   = 0

class Triangle:
    points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0]]) # 3x2
    vertex = torch.tensor([[0], [1], [2]]) # 3x1
    edge   = torch.tensor([[1, 2], [0, 2], [0, 1]]) # 3x2
    face   = tuple([[0, 1, 2]]) # 1x3
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = 0

class Quadrilateral:
    points = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]]) # 4x2
    vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
    edge   = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]) # 4x2
    face   = tuple([[0, 1, 2, 3]]) # 1x4
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = 0

class Tetrahedron:
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 4x3
    vertex = torch.tensor([[0], [1], [2], [3]]) # 4x1
    edge   = torch.tensor([[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]) # 6x2
    face   = tuple([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]) # 4x3
    cell   = torch.tensor([[0, 1, 2, 3]]) # 1x4
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

class Hexahedron:
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0],[1.0, 1.0, 1.0]]) # 8x3
    vertex = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7]]) # 8x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]) # 12x2
    face   = ([[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6], [1, 3, 5, 7], [2, 3, 6, 7], [4, 5, 6, 7]]) # 6x4
    cell   = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]) # 1x8
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

class Pyramid:
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[1.0, 1.0, 0.0],[0.0, 0.0, 1.0]]) # 5x3
    vertex = torch.tensor([[0], [1], [2], [3], [4]]) # 5x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]) # 8x2
    face   = tuple([[0, 1, 2, 3], [0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]) # 5x4
    cell   = torch.tensor([[0, 1, 2, 3, 4]]) # 1x5
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

class Prism:
    points = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[1.0, 0.0, 1.0],[0.0, 1.0, 1.0]]) # 5x3
    vertex = torch.tensor([[0], [1], [2], [3], [4], [5]]) # 6x1
    edge   = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]]) # 9x2
    face   = tuple([[0, 1, 2], [0, 1, 3, 4], [0, 2, 3, 5], [1, 2, 4, 5],[3,4,5]]) # 4x4
    cell   = torch.tensor([[0, 1, 2, 3, 4, 5]]) # 1x5
    dim    = points.shape[1]
    n_vertex = vertex.shape[0]
    n_edge   = edge.shape[0]
    n_face   = len(face)
    n_cell   = cell.shape[0]

