from layers.graph_layers import *
import layers.graph_layers
import layers.hybrid_gnn

# list of all possible layers
LAYERS = {
    'gcn'  : layers.graph_layers.GCN,
    'gat'  : layers.graph_layers.GAT,
    'mpnn' : layers.graph_layers.MPNN,
    'geo_quadrant_gcn' : layers.hybrid_gnn.GeoQuadrantGCN
}

def get(layer_name, default=None):
    layer = LAYERS.get(layer_name.lower(), default)
    
    if not layer:
        raise NotImplementedError(f'Layer with name "{layer_name}" is not implemented.')
    
    return layer