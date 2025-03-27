import torch

from direct import Direct
from loc_encoder import SirenNet


class Climplicit(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.location_encoder = SirenNet(
            dim_in=4,
            dim_hidden=512,
            dim_out=256,
            num_layers=16,
            dropout=False,
            h_siren=True,
            residual_connections=True)
        self.pos_embedding = Direct(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)

        self.location_encoder.load_state_dict(torch.load("climplicit.ckpt", weights_only=True))

        for name, param in self.location_encoder.named_parameters():
            param.requires_grad = False
        print("=> loaded Climplicit weights")

    def forward(self, coordinates, month=None):
        # Apply the positional embedding
        loc = self.pos_embedding(coordinates)

        if month is None:
            res = []
            # Get the Climplicit embeddings for four months across the year
            for m in [3, 6, 9, 12]:
                month = torch.ones(len(coordinates)) * m
                loc_month = torch.concat([loc, torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                          torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1)], dim=-1)
                res.append(self.location_encoder(loc_month))
            return torch.cat(res, dim=-1)

        # If we have a month
        # Append the month to the positional embedding
        loc_month = torch.concat([loc, torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                  torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1)], dim=-1)
        # Return the Climplicit embedding
        return self.location_encoder(loc_month)


if __name__ == "__main__":
    model = Climplicit()

    loc = [8.550155, 47.396702] # Lon/Lat or our office 
    april = 4                   # April
    batchsize = 10

    # Call with a month
    month = torch.ones(batchsize) * april
    print("Shape with month:", model(torch.tensor([loc] * batchsize), month).shape)

    # Call without month
    print("Shape without month:", model(torch.tensor([loc] * batchsize)).shape)
