
import argparse
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

import stylegan2
from stylegan2 import utils
from interpolation_utils import linear_interpolation, slerp_interpolation

#----------------------------------------------------------------------------

_description = """StyleGAN2 generator.
Run 'python %(prog)s <subcommand> --help' for subcommand help."""

#----------------------------------------------------------------------------

_examples = """examples:
  # Train a network or convert a pretrained one.
  # Example of converting pretrained ffhq model:
  python run_convert_from_tf --download ffhq-config-f --output G.pth D.pth Gs.pth

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate_images --network=Gs.pth --seeds=6600-6625 --truncation_psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate_images --network=Gs.pth --seeds=66,230,389,1518 --truncation_psi=1.0

  # Example of converting pretrained car model:
  python run_convert_from_tf --download car-config-f --output G_car.pth D_car.pth Gs_car.pth

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate_images --network=Gs_car.pth --seeds=6000-6025 --truncation_psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style_mixing_example --network=Gs.pth --row_seeds=85,100,75,458,1500 --col_seeds=55,821,1789,293 --truncation_psi=1.0
"""

#----------------------------------------------------------------------------

def _add_shared_arguments(parser):

    parser.add_argument(
        '--network',
        help='Network file path',
        required=True,
        metavar='FILE'
    )

    parser.add_argument(
        '--output',
        help='Root directory for run results. Default: %(default)s',
        type=str,
        default='./results',
        metavar='DIR'
    )

    parser.add_argument(
        '--pixel_min',
        help='Minumum of the value range of pixels in generated images. ' + \
            'Default: %(default)s',
        default=-1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--pixel_max',
        help='Maximum of the value range of pixels in generated images. ' + \
            'Default: %(default)s',
        default=1,
        type=float,
        metavar='VALUE'
    )

    parser.add_argument(
        '--gpu',
        help='CUDA device indices (given as separate ' + \
            'values if multiple, i.e. "--gpu 0 1"). Default: Use CPU',
        type=int,
        default=[],
        nargs='*',
        metavar='INDEX'
    )

    parser.add_argument(
        '--truncation_psi',
        help='Truncation psi. Default: %(default)s',
        type=float,
        default=0.5,
        metavar='VALUE'
    )


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=_description,
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    range_desc = 'NOTE: This is a single argument, where list ' + \
        'elements are separated by "," and ranges are defined as "a-b". Only integers are allowed.'

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    generate_images_parser = subparsers.add_parser(
        'generate_images', help='Generate images')

    generate_images_parser.add_argument(
        '--batch_size',
        help='Batch size for generator. Default: %(default)s',
        type=int,
        default=1,
        metavar='VALUE'
    )

    generate_images_parser.add_argument(
        '--seeds',
        help='List of random seeds for generating images. ' + range_desc,
        type=utils.range_type,
        default=None,
        metavar='RANGE'
    )

    generate_images_parser.add_argument(
        '--latents',
        help='List of paths to files with latent factors. ' + range_desc,
        type=str,
        default=None,
        metavar='DIR'
    )

    _add_shared_arguments(generate_images_parser)

    interpolate_parser = subparsers.add_parser(
        'interpolate', help='Interpolate images')

    interpolate_parser.add_argument(
        '--batch_size',
        help='Batch size for generator. Default: %(default)s',
        type=int,
        default=1,
        metavar='VALUE'
    )

    interpolate_parser.add_argument(
        '--seeds',
        help='List of random seeds for generating images. ' + range_desc,
        type=utils.range_type,
        default=None,
        metavar='RANGE'
    )

    interpolate_parser.add_argument(
        '--latents',
        help='List of paths to files with latent factors. ' + range_desc,
        type=str,
        default=None,
        metavar='DIR'
    )

    interpolate_parser.add_argument(
        '--number',
        help='Number of images.' + range_desc,
        type=int,
        required=True,
        metavar='VALUE'
    )

    interpolate_parser.add_argument(
        '--type',
        help='Interpolation type.',
        type=str,
        default='linear',
        required=False,
        metavar='DIR'
    )

    _add_shared_arguments(interpolate_parser)

    style_mixing_example_parser = subparsers.add_parser(
        'style_mixing_example', help='Generate style mixing video')

    style_mixing_example_parser.add_argument(
        '--row_seeds',
        help='List of random seeds for image rows. ' + range_desc,
        type=utils.range_type,
        required=True,
        metavar='RANGE'
    )

    style_mixing_example_parser.add_argument(
        '--col_seeds',
        help='List of random seeds for image columns. ' + range_desc,
        type=utils.range_type,
        required=True,
        metavar='RANGE'
    )

    style_mixing_example_parser.add_argument(
        '--style_layers',
        help='Indices of layers to mix style for. ' + \
            'Default: %(default)s. ' + range_desc,
        type=utils.range_type,
        default='0-6',
        metavar='RANGE'
    )

    style_mixing_example_parser.add_argument(
        '--grid',
        help='Save a grid as well of the style mix. Default: %(default)s',
        type=utils.bool_type,
        default=True,
        const=True,
        nargs='?',
        metavar='BOOL'
    )

    _add_shared_arguments(style_mixing_example_parser)

    return parser


#----------------------------------------------------------------------------

def set_device(G, args):
    """ Select appropriate device and copy the model on it
    """

    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)

    G.to(device)
    return device


def style_mixing_example(G, args):

    assert max(args.style_layers) < len(G), \
        'Style layer indices can not be larger than ' + \
        'number of style layers ({}) of the generator.'.format(len(G))

    # Set appropriate device
    device = set_device(G, args)

    G.static_noise()

    latent_size, label_size = G.latent_size, G.label_size
    G_mapping, G_synthesis = G.G_mapping, G.G_synthesis

    all_seeds = list(set(args.row_seeds + args.col_seeds))
    all_z = torch.stack([torch.from_numpy(np.random.RandomState(seed).randn(latent_size)) for seed in all_seeds])
    all_z = all_z.to(device=device, dtype=torch.float32)
    if label_size:
        labels = torch.zeros(len(all_z), dtype=torch.int64, device=device)
    else:
        labels = None

    print('Generating disentangled latents...')
    with torch.no_grad():
        all_w = G_mapping(latents=all_z, labels=labels)

    all_w = all_w.unsqueeze(1).repeat(1, len(G_synthesis), 1)

    w_avg = G.dlatent_avg

    if args.truncation_psi != 1:
        all_w = w_avg + args.truncation_psi * (all_w - w_avg)

    w_dict = {seed: w for seed, w in zip(all_seeds, all_w)}

    all_images = []

    progress = utils.ProgressWriter(len(all_w))
    progress.write('Generating images...', step=False)

    with torch.no_grad():
        for w in all_w:
            all_images.append(G_synthesis(w.unsqueeze(0)))
            progress.step()

    progress.write('Done!', step=False)
    progress.close()

    all_images = torch.cat(all_images, dim=0)

    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, all_images)}

    progress = utils.ProgressWriter(len(args.row_seeds) * len(args.col_seeds))
    progress.write('Generating style-mixed images...', step=False)

    for row_seed in args.row_seeds:
        for col_seed in args.col_seeds:
            w = w_dict[row_seed].clone()
            w[args.style_layers] = w_dict[col_seed][args.style_layers]
            with torch.no_grad():
                image_dict[(row_seed, col_seed)] = G_synthesis(w.unsqueeze(0)).squeeze(0)
            progress.step()

    progress.write('Done!', step=False)
    progress.close()

    progress = utils.ProgressWriter(len(image_dict))
    progress.write('Saving images...', step=False)

    for (row_seed, col_seed), image in list(image_dict.items()):
        image = utils.tensor_to_PIL(
            image, pixel_min=args.pixel_min, pixel_max=args.pixel_max)
        image_dict[(row_seed, col_seed)] = image
        image.save(os.path.join(args.output, '%d-%d.png' % (row_seed, col_seed)))
        progress.step()

    progress.write('Done!', step=False)
    progress.close()

    if args.grid:
        print('\n\nSaving style-mixed grid...')
        H, W = all_images.size()[2:]
        canvas = Image.new(
            'RGB', (W * (len(args.col_seeds) + 1), H * (len(args.row_seeds) + 1)), 'black')
        for row_idx, row_seed in enumerate([None] + args.row_seeds):
            for col_idx, col_seed in enumerate([None] + args.col_seeds):
                if row_seed is None and col_seed is None:
                    continue
                key = (row_seed, col_seed)
                if row_seed is None:
                    key = (col_seed, col_seed)
                if col_seed is None:
                    key = (row_seed, row_seed)
                canvas.paste(image_dict[key], (W * col_idx, H * row_idx))
        canvas.save(os.path.join(args.output, 'grid.png'))
        print('Done!')

#----------------------------------------------------------------------------

def build_latents_names(type_, G, args):
    """ type_ is 'generate' or 'interpolate'
    """

    latents, names = [], []

    if args.seeds:
        # Generate the latent factors from the seeds
        for seed in args.seeds:
            rnd = np.random.RandomState(seed)
            latents.append(torch.from_numpy(rnd.randn(G.latent_size)).type(torch.float32))
            names.append('seed_%04d' % seed)
        latents = torch.stack(latents, dim=0)

    if args.latents:
        # Read the files with the latent factors
        for e, l in enumerate(args.latents.split(',')):
            latents.append(torch.load(l))
            names.append('latent_%04d' % e)
        latents = torch.cat(latents, dim=0)

    if type_=='generate':
        print(latents)
        return latents, names

    elif type_=='interpolate':
        assert len(latents)==2, '2 latents should be provided when interpolating.'

        print('Interpolating latents factors...')
        if args.type=='linear':
            print('Linear interpolation')
            latents_interpolated = linear_interpolation(latents[0].numpy(), latents[1].numpy(), args.number)
        elif args.type=='slerp':
            print('Slerp interpolation')
            latents_interpolated = slerp_interpolation(latents[0].numpy(), latents[1].numpy(), args.number)

        latents_interpolated = torch.from_numpy(latents_interpolated).unsqueeze(1)
        return latents_interpolated, ['interpolate_%04d' % i for i in range(args.number + 1)]


def generate_images(type_, G, args):

    device = set_device(G, args)

    # Set truncation_psi
    if args.truncation_psi != 1:
        G.set_truncation(truncation_psi=args.truncation_psi)

    # Set random noise (instead of noise generated from seed)
    G.random_noise()

    # Get labels (deactivated for the moment)
    labels = None
    #if G.label_size:
    #    labels = [torch.tensor([np.random.randint(0, G.label_size)]) for _ in range(nb_images)]
    #    labels = torch.cat(labels, dim=0).to(device=device, dtype=torch.int64)

    # Get latent factors and names of images
    latents, names = build_latents_names(type_, G, args)
    latents.to(device=device, dtype=torch.float32)

    # Generate images per batch
    for i in tqdm(range(0, latents.shape[0], args.batch_size)):
        batch_latents = latents[i: i + args.batch_size]
        batch_names = names[i: i + args.batch_size]

        with torch.no_grad():
            if args.seeds:
                # In this case the latents factors must be fed first into the mapping network
                generated = G(batch_latents, labels=labels)
            if args.latents:
                # In this case we assume that the latent factors are output of mapping network (because that's the output of the projection)
                # Labels are not handled here
                generated = G.G_synthesis(latents=batch_latents)

        images = utils.tensor_to_PIL(generated, pixel_min=args.pixel_min, pixel_max=args.pixel_max)
        for name, img in zip(batch_names, images):
            img.save(os.path.join(args.output, name + '.png'))


#----------------------------------------------------------------------------

def main():

    args = get_arg_parser().parse_args()

    assert args.command, 'Missing subcommand.'
    assert os.path.isdir(args.output) or not os.path.splitext(args.output)[-1], \
        '--output argument should specify a directory, not a file.'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    G = stylegan2.models.load(args.network)
    G.eval()

    assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
        'stylegan2.models.Generator. Found {}.'.format(type(G))

    if args.command == 'generate_images':
        generate_images('generate', G, args)
    elif args.command == 'interpolate':
        generate_images('interpolate', G, args)
    elif args.command == 'style_mixing_example':
        style_mixing_example(G, args)
    else:
        raise TypeError('Unkown command {}'.format(args.command))

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
