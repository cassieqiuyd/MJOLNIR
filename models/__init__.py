from .basemodel import BaseModel
from .gcn import GCN
from .savn import SAVN
from .kgvn import KGVN
from .kgvn1 import KGVN1
from .mjolnir_r import KGVN2
from .mjolnir_o import KGVN3
from .kgvn4 import KGVN4

__all__ = ["BaseModel", "GCN", "SAVN", "KGVN", "KGVN1", "KGVN2","KGVN3","KGVN4"]

variables = locals()
