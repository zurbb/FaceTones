
import torch.nn as nn
import torch
import torch.nn.functional as F
# from sklearn.metrics.pairwise import cosine_similarity

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
            # self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # output 12,65,192
            nn.Conv2d(8, 2, kernel_size=3, stride=2, padding=1),  # output 2,33,96
            nn.ReLU(), 
            # self.dropout,
            nn.Conv2d(2, 1, kernel_size=3, stride=2, padding=1),  # output 1,17,48
            nn.ReLU(), 
            # self.dropout,
            nn.Flatten(),  # output 1,816
        )
        self.multihead = nn.MultiheadAttention(embed_dim=816, num_heads=8) 
        self.final_layer = nn.Linear(816, 512)  # output 1,768
        # self.loss_func = CosineTripletLoss(margin=0.8)
        # self.loss_func = ContrastiveCosineLoss()
        self.loss_func = CrossEntropyCosineLoss()
        
    def forward(self, x):
        logits = self.convolution_layers(x.to(device))
        attn_output, _ = self.multihead(logits.to(device), logits.to(device), logits.to(device), need_weights=False)
        # attn_output = self.dropout(attn_output.to(device))  # Apply dropout
        logits = self.final_layer(attn_output.to(device))
        return logits
    
    def loss(self,outputs, voices):
        voices = voices.to(outputs.device)
        loss = self.loss_func(outputs, voices)
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


class ContrastiveCosineLoss(nn.Module):
    def __init__(self):
        super(ContrastiveCosineLoss, self).__init__()
        self.k = 20

    def forward(self, outputs, voices):
        sim_vector = F.cosine_similarity(outputs, voices)
        numerator = torch.exp(sim_vector)
        sum_negatives = torch.zeros(outputs.size(0)).to(outputs.device)
        for i in range(1, self.k+1):
            shifted_voices = torch.roll(voices, shifts=i, dims=0)
            sum_negatives += torch.exp(F.cosine_similarity(outputs, shifted_voices))

        denominator = numerator + sum_negatives
        log_sim_loss = -torch.log(numerator / denominator)
        # loss = 1 - numerator / denominator
        return torch.mean(log_sim_loss)

class CrossEntropyCosineLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyCosineLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, voices):
        # like in clip
        outputs = F.normalize(outputs, p=2, dim=1)
        voices = F.normalize(voices, p=2, dim=1)
        logits = torch.tensordot(outputs, voices.T, dims=1) # simialrities, [n,n]
        diagonal_mask = torch.eye(outputs.size(0)).to(outputs.device)
        off_diagonal_mask = torch.clamp(logits - 0.8, min=0)
        masked_logits = logits * diagonal_mask + off_diagonal_mask * (1-diagonal_mask)
        labels = torch.arange(outputs.size(0)).to(outputs.device)
        axis_1 = self.loss(masked_logits, labels)
        axis_2 = self.loss(masked_logits.T, labels)
        return (axis_1 + axis_2) / 2 



