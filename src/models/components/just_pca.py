import torch
from sklearn.decomposition import PCA
import numpy as np

class PCAEncoder(torch.nn.Module):
    """Take PCA to (11) dim for the (128) units vectors."""
    def __init__(self, input_len=11, output_dim=128):
        super().__init__()
        
        self.transform = PCA(input_len).fit_transform(np.eye(output_dim))
        self.transform = torch.tensor(self.transform).t().float()
        # Scaling to normal distribution for numerical stability
        self.transform = (self.transform - self.transform.mean()) / self.transform.std()
        
    def forward(self, x):
        self.transform = self.transform.to(x.device)
        return x @ self.transform



if __name__ == "__main__":
    enc = PCAEncoder()
    print(enc(torch.ones((2048, 11))).shape)
