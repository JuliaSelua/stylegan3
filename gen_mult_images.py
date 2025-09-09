# gen_id_images.py
import os
import re
import click
import pickle
import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy

# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=int, multiple=True, help='List of seeds (1 pro ID)')
@click.option('--n_styles', type=int, default=5, help='Number of style variations per ID')
@click.option('--outdir', type=str, required=True, metavar='DIR', help='Where to save the output images')

def generate_images(network_pkl: str, seeds: tuple, n_styles: int, outdir: str):
    """Generate images from a trained network pickle with multiple images per ID."""

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    for seed in seeds:
        print(f'Generating ID {seed}...')
        torch.manual_seed(seed)
        np.random.seed(seed)

        #z_id = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        z_id = torch.from_numpy(np.random.randn(1, G.z_dim_id)).to(device)


        for style_idx in range(n_styles):
            #z_style = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            z_style = torch.from_numpy(np.random.randn(1, G.z_dim_style)).to(device)

            img = G(z_id, None, z2=z_style)
            img = (img.clamp(-1,1) + 1) * 127.5
            img = img.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)[0]

            # save as id{seed}_style{style_idx}.png
            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/id{seed}_style{style_idx}.png')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
