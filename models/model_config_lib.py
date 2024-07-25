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
    """
    A class to represent the model that gets an image and returns an embedding
    matching the voice of the person in the image.
    """
    
    def __init__(self):
        super().__init__()
        # network architecture
        self.linear_seq = nn.Sequential(
            nn.Linear(768, 512), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512), 
            nn.Linear(512, 512),  
            nn.AvgPool2d(kernel_size=2, stride=4, ceil_mode=True),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(65 * 128, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.LayerNorm(1024), 
            nn.GELU(),
            nn.Linear(1024, 512)
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
    """
    Our first attempt at a contrastive loss function.
    """
    def __init__(self, margin):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = nn.CosineEmbeddingLoss(anchor, positive, torch.ones(anchor.size(0)).to(anchor.device)) # close ==> 0
        neg_distance = nn.CosineEmbeddingLoss(anchor, negative, torch.ones(anchor.size(0)).to(anchor.device)) # far ==> 1
        losses = F.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()


class ContrastiveCosineLoss(nn.Module):
    """
    Our second attempt at a contrastive loss function.
    """
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
    """
    Our third attempt at a contrastive loss function.
    This is the one we currently use.
    Based somewhat on the implementation in CLIP.
    We added a margin, to make the model not try to make the similarities of negative examples too low
    (because in reality, different voices are not that different in our embedding space).
    We made this margin learnable, to find the best value for it, but clamp it because otherwise it goes to 1.
    we might get better results with different maximum clamping values (or even fixed values).
    Should be tested further.
    """
    def __init__(self):
        super(CrossEntropyCosineLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.learnable_param = nn.Parameter(torch.tensor(0.7))
        self.positive_mean_loss = 0
        self.entropy_loss = 0
    
    def forward(self, outputs, voices):
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
        self.entropy_loss =(axis_1 + axis_2) / 2
        self.positive_mean_loss = 1 - torch.diagonal(logits, offset=0).to(outputs.device).float().mean()
        return  self.entropy_loss + self.positive_mean_loss