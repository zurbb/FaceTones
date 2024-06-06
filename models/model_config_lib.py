import torch.nn as nn

class ImageToVoice(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # Dropout layer
        self.convolution_layers = nn.Sequential(
            #input 1,257,768
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # output 8,129,384
            nn.ReLU(),
            #self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # output 12,65,192
            nn.Conv2d(8, 2, kernel_size=3, stride=2, padding=1),  # output 2,33,96
            nn.ReLU(), 
            #self.dropout,
            nn.Conv2d(2, 1, kernel_size=3, stride=2, padding=1),  # output 1,17,48
            nn.ReLU(), 
            self.dropout,
            nn.Flatten(),  # output 1,816
        )
        self.multihead = nn.MultiheadAttention(embed_dim=816, num_heads=8) 
        self.self.final_layer = nn.Linear(816, 512)  # output 1,768
        self.loss = CosineTripletLoss(margin=0.8)
        
    def forward(self, x):
        logits = self.convolutional_layers(x.to(device))
        attn_output, _ = self.multihead(logits.to(device), logits.to(device), logits.to(device), need_weights=False)
        attn_output = self.dropout(attn_output.to(device))  # Apply dropout
        logits = self.final_layer(attn_output.to(device))
        return logits
    
    def loss(outputs, voices):
        
        voices = voices.to(outputs.device)
        anchor = outputs
        positive = voices
    
        shift = torch.randint(1, voices.size(0), (1,)).item()
        indices = torch.arange(voices.size(0)).to(outputs.device)
        indices = (indices + shift) % voices.size(0)
        negative = voices[indices]

        loss = self.loss(anchor, positive, negative)
    
        return loss
        


class CosineTripletLoss(nn.Module):
    def __init__(self, margin):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = nn.CosineEmbeddingLoss(anchor, positive, torch.ones(anchor.size(0)).to(anchor.device)) # close ==> 0
        neg_distance = nn.CosineEmbeddingLoss(anchor, negative, torch.ones(anchor.size(0)).to(anchor.device)) # far ==> 1
        losses = F.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()

