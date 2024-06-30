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
                # Vision network (VGG-inspired CNN)
        self.vision_cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # output: 64, 257, 768
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # output: 64, 257, 768
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: 64, 128, 384
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # output: 128, 128, 384
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # output: 128, 128, 384
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: 128, 64, 192
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # output: 256, 64, 192
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # output: 256, 64, 192
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # output: 256, 64, 192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: 256, 32, 96
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # output: 512, 32, 96
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # output: 512, 32, 96
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # output: 512, 32, 96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: 512, 16, 48
            
            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # output: 512, 16, 48
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # output: 512, 16, 48
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # output: 256, 16, 48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: 256, 8, 24
            
            # Flatten
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(256 * 8 * 24, 512),  # output: 512
            nn.ReLU(),
            nn.Dropout(0.2),
     
        )
        self.loss_func = CrossEntropyCosineLoss()
        
    def forward(self, x):
        logits = self.vision_cnn(x.to(device))
        # attn_output, _ = self.multihead(logits.to(device), logits.to(device), logits.to(device), need_weights=False)
        # attn_output = self.dropout(attn_output.to(device))  # Apply dropout
        # logits = self.final_layer(attn_output.to(device))
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


