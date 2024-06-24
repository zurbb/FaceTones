import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if __name__ == "__main__":
    # n = 5
    # # Example of target with class indices
    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(n, n, requires_grad=True)
    # target = torch.Tensor(np.arange(n))
    # output = loss(input, target)
    # output.backward()
    # print(f"CrossEntropyLoss: {output}")
    
    model =  lib.load_model_by_checkpoint(f"{args.model_checkpoint}")
    model.eval()
    with torch.inference_mode():
        print("Loading validation data")
        batch_size = args.batch_size