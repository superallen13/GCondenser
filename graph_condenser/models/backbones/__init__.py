from .appnp import APPNP
from .cheb import ChebNet
from .gat import GAT
from .gcn import GCN
from .sage import SAGE
from .sgc import SGC
from .mlp import MLP


def load_backbone(backbone_name: str):
    """
    A factory function to load the backbone model
    """
    if backbone_name == "appnp":
        return APPNP
    elif backbone_name == "chebnet":
        return ChebNet
    elif backbone_name == "gat":
        return GAT
    elif backbone_name == "gcn":
        return GCN
    elif backbone_name == "sage":
        return SAGE
    elif backbone_name == "sgc":
        return SGC
    elif backbone_name == "mlp":
        return MLP
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
