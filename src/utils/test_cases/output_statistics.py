import torch

from src.utils.test_cases.switzerland_dataset import *


class OutputStatistics:
    def __init__(self, month, use_training_coordinates=False):
        self.month = month
        self.use_training_coordinates = use_training_coordinates

    def __call__(self, pos_embedding, location_encoder, wb, section="test/"):
        if self.use_training_coordinates:
            ds = SwitzerlandDatasetTC()
        else:
            ds = SwitzerlandDataset()
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=8196,
            num_workers=16,
            shuffle=False,
        )

        encodings = []
        for lonlat in dl:
            # Get pos embedding
            loc = pos_embedding(lonlat)
            # Put on the same device
            loc = loc.squeeze(dim=1).to("cuda")
            # Append month
            month = torch.full([len(loc)], self.month).to("cuda")
            loc_month = torch.concat(
                [
                    loc,
                    torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                    torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                ],
                dim=-1,
            )
            # Get encoding from network
            with torch.no_grad():
                encodings.append(location_encoder(loc_month))

        encodings = torch.concat(encodings, dim=0).cpu().numpy()

        wb.log(
            {
                section + "output max": encodings.max(axis=-1),
                section + "output min": encodings.min(axis=-1),
                section + "output mean": encodings.mean(axis=-1),
            }
        )
