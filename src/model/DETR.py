import torch
import torch.nn as nn

from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50().children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        
        # [H x W, b, hidden_dim]
        pos_embed =torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),    # [W, 128] → [1, W, 128] → [H, W, 128]
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),    # [H, 128] → [H, 1, 128] → [H, W, 128]
        ], dim=-1).flatten(0, 1).unsqueeze(1)                   # [H, W, 256] → [HxW, 256] → [HxW, 1, 128]
        
        print(pos_embed.shape)
        print(h.shape)
        print(h.flatten(2).shape)
        print(h.flatten(2).permute(2, 0, 1).shape)
        print(self.query_pos.unsqueeze(1).shape)
        
        # encoder input shape: [image_token_num, baich_size, hidden_dim]
        # decoder input shape: [bbox_num, batch_size, hidden_dim]
        h =self.transformer(pos_embed + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
    
    
if __name__ == '__main__':
    detr = DETR(91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
    detr.eval()
    inputs = torch.randn(1, 3, 320, 480)
    logits, bboxes = detr(inputs)
