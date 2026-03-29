# gen_id_images.py
import os
import click
import pickle
import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy
from torchvision.utils import save_image, make_grid

# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=int, multiple=True, help='List of seeds (1 pro ID)')
@click.option('--n_styles', type=int, default=25, help='Number of style variations per ID (muss Quadrat sein für Grid, z.B. 25=5x5)')
@click.option('--outdir', type=str, required=True, metavar='DIR', help='Where to save the output grids')

def generate_images(network_pkl: str, seeds: tuple, n_styles: int, outdir: str):
    """Generate iDiff-compatible identity grids from a trained StyleGAN network pickle."""

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    grid_size = int(np.sqrt(n_styles))
    assert grid_size * grid_size == n_styles, \
        f"--n_styles {n_styles} muss ein Quadrat sein (z.B. 25 für 5x5)"

    for seed in seeds:
        print(f'Generating ID {seed}...')
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ID-Latent (fixiert)
        z_id = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

        images = []
        for style_idx in range(n_styles):
            # Style-Latent (variiert)
            z_style = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

            img = G(z_id, None, z2=z_style)
            img = (img.clamp(-1,1) + 1) / 2  # [0,1] für save_image
            images.append(img)

        images = torch.cat(images, dim=0)

        # Grid erstellen
        grid = make_grid(images, nrow=grid_size, padding=0)

        filename = os.path.join(outdir, f"id{seed:05d}.png")
        save_image(grid, filename)
        print(f"Saved {filename}")

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
