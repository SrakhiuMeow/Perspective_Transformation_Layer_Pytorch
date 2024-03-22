import torch
import torch.nn as nn
from pers_layer import *

class InsertLayerModel(nn.Module):
    def __init__(self, insert_id, num_filters, base_model=None, random=False):
        super(InsertLayerModel, self).__init__()
        self.base_model = base_model
        self.insert_id = insert_id
        self.num_filters = num_filters
        self.random = random
        
        self.model = []
          
        for i, layer in enumerate(self.base_model.children()):
            self.model.append(layer)
            if i == self.insert_id:
                if self.num_filters == 1:
                    pt = Perspective_Layer()
                    self.model.append(pt)
                else:
                    pt = Perspective_Layer(tm=self.num_filters, random_init=self.random)
                    conv = nn.Conv2d(layer.out_channels*self.num_filters, layer.out_channels, 1)
                    self.model.append(pt)
                    self.model.append(conv)
              
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x

def insert_layer(insert_id, num_filters, model, random=False):
    """
    Insert a layer into the model.

    Parameters:
    - insert_id (int): The ID of the layer to be inserted.
    - num_filters (int): The number of filters (perspective transformation matrix) for the inserted layer.
    - model (Model): The model to insert the layer into.
    - random (bool): Whether to initialize the layer with random weights. Default is False.

    Returns:
    - insert_layer_model (InsertLayerModel): The model with the inserted PT layer.
    """
    insert_layer_model = InsertLayerModel(insert_id, num_filters, model, random)
    return insert_layer_model

