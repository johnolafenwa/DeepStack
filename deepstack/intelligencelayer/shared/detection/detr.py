import torch 
import torch.nn as nn 
from torchvision.models import resnet50 


class DETR(nn.Module):

    def __init__(self, num_classes: int, hidden_dim: int = 256, nheads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6):

        super().__init__()

        self.backbone = resnet50()
        del self.backbone.fc 

        #create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        #create a transformer layer 
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        self.linear_class = nn.Linear(hidden_dim,num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        #output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        #spatial positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs: torch.Tensor):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ], dim=-1).flatten(0,1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
        self.query_pos.unsqueeze(1)).transpose(0,1)

        # project transformer outputs to class labels and bounding boxes

        return self.linear_class(h), self.linear_bbox(h).sigmoid()