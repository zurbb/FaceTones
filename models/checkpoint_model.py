import torch.nn as nn
import torch
import torch.nn.functional as F

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
class CheckPointImageToVoice(nn.Module):
    """
    This class defines the model architecture for the image-to-voice checkpoint model used in the game.
    It is a copy of the last most successful model trained in the training notebook.
    """
    
    def __init__(self):
        super().__init__()
        self.linear_seq = nn.Sequential(
            nn.Linear(768, 512), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512), 
            nn.Linear(512, 512),  
            nn.AvgPool2d(kernel_size=2, stride=4, ceil_mode=True),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(65 * 128, 512) 
        )
        self.loss_func = CrossEntropyCosineLoss()
        
    def forward(self, x):
        logits = self.linear_seq(x.to(device))
      
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
        self.learnable_param = nn.Parameter(torch.tensor(0.7))
    
    def forward(self, outputs, voices):
        # like in clip
        outputs = F.normalize(outputs, p=2, dim=1)
        voices = F.normalize(voices, p=2, dim=1)
        logits = torch.tensordot(outputs, voices.T, dims=1) # simialrities matrix, [n,n]
        diagonal_mask = torch.eye(outputs.size(0)).to(outputs.device)
        margin = torch.clamp(self.learnable_param,max=0.9,min=0.7)
        off_diagonal_mask = torch.clamp(logits - margin, min=0)
        masked_logits = logits * diagonal_mask + off_diagonal_mask * (1-diagonal_mask)
        labels = torch.arange(outputs.size(0)).to(outputs.device)
        axis_1 = self.loss(masked_logits, labels)
        axis_2 = self.loss(masked_logits.T, labels)
        entropy_loss =(axis_1 + axis_2) / 2
        postive_mean_loss =1 - torch.diagonal(logits, offset=0).to(outputs.device).float().mean()
        return  entropy_loss + postive_mean_loss


