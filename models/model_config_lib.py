
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
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
        self.final_layer = nn.Linear(816, 512)  # output 1,768
        self.loss_func = CosineTripletLoss(margin=0.8)
        
    def forward(self, x):
        logits = self.convolutional_layers(x.to(device))
        attn_output, _ = self.multihead(logits.to(device), logits.to(device), logits.to(device), need_weights=False)
        attn_output = self.dropout(attn_output.to(device))  # Apply dropout
        logits = self.final_layer(attn_output.to(device))
        return logits
    
    def loss(self,outputs, voices):
        
        voices = voices.to(outputs.device)
        anchor = outputs
        positive = voices
    
        shift = torch.randint(1, voices.size(0), (1,)).item()
        indices = torch.arange(voices.size(0)).to(outputs.device)
        indices = (indices + shift) % voices.size(0)
        negative = voices[indices]

        loss = self.loss_func(anchor, positive, negative)
    
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


class ContrastiveCoiseLoss(nn.Module):
    def __init__(self):
        super(ContrastiveCoiseLoss, self).__init__()


# take all outputs and calculate the similarty between them and the voices ->sim_matrix 
# for each outputs i:
# loss = 1 - exp(sim_matirx[i])/(sum(exp(sim_matrix[j])) + sim_matrix[i]) // j!=i
# the overall loss is the mean of all losses  
    def forward(self, outputs, voices):
        sim_matrix = cosine_similarity(outputs, voices)
        losses = []
        for i in range(outputs.size(0)):
            numerator = torch.exp(sim_matrix[i])
            denominator = torch.sum(torch.exp(sim_matrix)) + sim_matrix[i]
            loss = 1 - numerator / denominator
            losses.append(loss)
        loss = torch.mean(torch.stack(losses))
        return loss
