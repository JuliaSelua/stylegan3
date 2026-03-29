# gen_id_images_individual.py
import os
import click
import torch
import numpy as np
import PIL.Image
import legacy

# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=int, multiple=True, help='List of seeds (1 pro ID)')
@click.option('--n_styles', type=int, default=25, help='Number of style variations per ID')
@click.option('--outdir', type=str, required=True, metavar='DIR', help='Where to save the output images')

def generate_images(network_pkl: str, seeds: tuple, n_styles: int, outdir: str):
    """Generate individual images for each ID and style from a trained network pickle."""

    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    for seed in seeds:
        print(f'Generating ID {seed}...')
        torch.manual_seed(seed)
        np.random.seed(seed)

        z_id = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        id_dir = os.path.join(outdir, f"id{seed:05d}")
        os.makedirs(id_dir, exist_ok=True)

        for style_idx in range(n_styles):
            z_style = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

            img = G(z_id, None, z2=z_style)
            img = (img.clamp(-1,1) + 1) * 127.5
            img = img.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)[0]

            filename = os.path.join(id_dir, f"id{seed:05d}_style{style_idx:02d}.png")
            PIL.Image.fromarray(img, 'RGB').save(filename)

        print(f"Saved {n_styles} images for ID {seed}")

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()
