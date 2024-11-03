import os
import re
import toml 
from .element import Element,\
                     Line,\
                     Triangle, \
                     Quadrilateral, \
                     Tetrahedron, \
                     Hexahedron, \
                     Prism, \
                     Pyramid
from typing import Callable,Dict,List,Type

from .element_type2dimension import element_type2dimension
from .element_type2order import element_type2order

element_types:List[str] = list(element_type2dimension.keys())

def element_type2element(x:str)->Type[Element]:
    element_prefix = re.findall(r'[a-zA-Z]+', x)[0]
    return {
        'line' : Line,
        'triangle' : Triangle, 
        'quad' : Quadrilateral,
        'tetra' : Tetrahedron,
        'hexahedron' : Hexahedron,
        'pyramid' : Pyramid,
        'wedge' : Prism
    }[element_prefix]

