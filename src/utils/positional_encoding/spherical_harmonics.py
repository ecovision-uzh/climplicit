import torch
from torch import nn

try:
    from spherical_harmonics_ylm import SH as SH_analytic
except:
    from src.utils.positional_encoding.spherical_harmonics_ylm import SH as SH_analytic

"""
Spherical Harmonics location encoder.
Taken from https://github.com/microsoft/satclip/tree/main/satclip/positional_encoding
"""


class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        self.SH = SH_analytic

    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l in range(0, self.L):
            for m in range(-l, l + 1):
                y = self.SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)
                if torch.isnan(y).sum() > 0:
                    print(phi, theta, phi.dtype, theta.dtype, l, m, y)
                    raise ValueError()
        return torch.stack(Y, dim=-1).float()


if __name__ == "__main__":
    from tqdm import tqdm

    BS, EXPS, POLYS = (32000, 10, 41)
    print("Testing for BS", BS, "- EXPS", EXPS, "- POLYS", POLYS)
    sh = SphericalHarmonics(legendre_polys=POLYS)
    lonlat = torch.rand([BS, 2]) * 180 - 90  # Random values in [-90,90],[-90,90]
    for i in tqdm(range(EXPS)):
        out = sh(lonlat)
    print(out[0], len(out[0]))
