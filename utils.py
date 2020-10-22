
import os
from subprocess import Popen
import shlex
from PIL import Image
import copy
import torch


DEFAULT_TRUNCATION_PSI = 0.5

#########
# Run commands
#########

def run_generate(model_path, out_path, seeds=None, latents=None, truncation_psi=DEFAULT_TRUNCATION_PSI):
    """
    with higher psi, you can get higher diversity on the generated images but it also has a higher chance of generating weird or broken faces
    """

    seeds_or_latents = ','.join(map(str, seeds)) if seeds else ','.join(latents)
    seeds_or_latents_arg = f'--seeds={seeds_or_latents}' if seeds else f'--latents={seeds_or_latents}'

    os.makedirs(out_path, exist_ok=True)
    cmd = f'python run_generator.py generate_images --network={model_path} {seeds_or_latents_arg} --output={out_path} --truncation_psi={truncation_psi}'
    Popen(shlex.split(cmd)).wait()


def run_interpolate(model_path, out_path, number, seeds=None, latents=None, truncation_psi=DEFAULT_TRUNCATION_PSI, type_=None):

    seeds_or_latents = ','.join(map(str, seeds)) if seeds else ','.join(latents)
    seeds_or_latents_arg = f'--seeds={seeds_or_latents}' if seeds else f'--latents={seeds_or_latents}'
    type_arg = f'--type={type_}' if type_ else ''

    os.makedirs(out_path, exist_ok=True)
    cmd = f'python run_generator.py interpolate --network={model_path} {seeds_or_latents_arg} --output={out_path} --number={number} --truncation_psi={truncation_psi} {type_arg}'
    Popen(shlex.split(cmd)).wait()


def run_metrics(model_path, out_path, data_dir, num_samples, gpu=True):
    
    os.makedirs(out_path, exist_ok=True)
    gpu_str = '--gpu=0' if gpu else ''
    cmd = f'python run_metrics.py fid --network={model_path} --num_samples={num_samples} --output={out_path} --data_dir={data_dir} {gpu_str}'
    Popen(shlex.split(cmd)).wait()


def run_projection(model_path, out_path, input_path, num_steps=1000, num_snapshots=5, gpu=True):

    os.makedirs(out_path, exist_ok=True)
    gpu_str = '--gpu=0' if gpu else ''
    cmd = f'python run_projector.py project_real_images --network={model_path} --num_steps={num_steps} --num_snapshots={num_snapshots} --output={out_path} --data_dir={input_path} {gpu_str}'
    Popen(shlex.split(cmd)).wait()



#########
# Image utilities
#########


def build_image(input_paths, output_path, nb_rows=1):
    """
    Build an image concatenating all images specified with input_paths
    All iamges have the same resolution.
    nb_rows is the number of rows in the output image.
    """

    width, height = Image.open(f'{input_paths[0]}').size
    nb_cols = 1 + (len(input_paths) - 1) // nb_rows
    new_im = Image.new('RGB', (width * nb_cols, height * nb_rows))

    for e, i in enumerate(input_paths):
        image = Image.open(i)
        row, col = e // nb_cols, e - nb_cols * (e // nb_cols)
        new_im.paste(image, (col * width, row * height))

    new_im.save(output_path)





#########
# Final functions
#########


def model_learning(start_model_path, checkpoints_path, output_path, seeds, truncation_psi=DEFAULT_TRUNCATION_PSI, sampling=1):
    """
    Generate images at each checkpoint found
    """

    # Generate images from start model
    run_generate(start_model_path, f'{output_path}/0', seeds)

    # List the available checkpoints
    checkpoints = sorted(os.listdir(checkpoints_path), key=lambda x: int(x.split('_')[0]))
    checkpoints = [c for e, c in enumerate(checkpoints) if e % sampling == 0]
    for e, c in enumerate(checkpoints):
        run_generate(f'{checkpoints_path}/{c}/Gs.pth', f'{output_path}/{e + 1}', seeds, truncation_psi)

    # Build image
    all_paths = []
    for s in seeds:
        s_path = 'seed%04d.png' % s
        for e in range(len(checkpoints) + 1):
            all_paths.append(f'{output_path}/{e}/{s_path}')

    build_image(all_paths, f'{output_path}/evaluation.png', nb_rows=len(seeds))


def compute_metrics(start_model_path, checkpoints_path, output_path, data_dir, num_samples=10000, sampling=1, gpu=True):
    """
    num_samples: 10k looks OK to compute the FID, however it is a bit higher than when using 50k samples
    """

    # List all models being evaluated
    checkpoints = sorted(os.listdir(checkpoints_path), key=lambda x: int(x.split('_')[0]))
    checkpoints = [c for e, c in enumerate(checkpoints) if (e + 1) % sampling == 0]
    models = [start_model_path] + [f'{checkpoints_path}/{c}/Gs.pth' for c in checkpoints]

    # Run the metrics
    for e, m in enumerate(models):
        print(f'Evaluating model {m}')
        run_metrics(m, f'{output_path}/{e}', data_dir, num_samples, gpu=True)



###
# Load a new model (generator and discriminator) with an increased resolution
# Set the parameters from a lower-resolution model
###

def build_model(g_model_path, d_model_path, channels):

    # Read models
    g_model = torch.load(g_model_path)
    d_model = torch.load(d_model_path)

    # Set new channels
    old_channels = g_model['G_synthesis']['kwargs']['channels']
    g_model['G_synthesis']['kwargs']['channels'] = channels
    d_model['kwargs']['channels'] = channels

    # Parameters to add in generator are, for each new channel:
    # - conv_blocks.{n}.conv_block.{0|1}.bias == tensor(channels[-n-1]) with 0
    # - conv_blocks.{n}.conv_block.{0|1}.layer.weight == tensor(1) with 0
    # - conv_blocks.{n}.conv_block.0.layer.layer.dense.layer.weight == tensor(channels[-n], 512) random
    # - conv_blocks.{n}.conv_block.1.layer.layer.dense.layer.weight == tensor(channels[-n-1], 512) random
    # - conv_blocks.{n}.conv_block.0.layer.layer.weight == tensor(channels[-n-1], channels[-n], 3, 3) random
    # - conv_blocks.{n}.conv_block.1.layer.layer.weight == tensor(channels[-n-1], channels[-n-1], 3, 3) random
    # - conv_blocks.{n}.conv_block.0.layer.layer.dense.bias == tensor(channels[-n]) with 1
    # - conv_blocks.{n}.conv_block.1.layer.layer.dense.bias == tensor(channels[-n-1]) with 1
    #
    # - to_data_layers.{n}.bias == tensor(3) with 0
    # - to_data_layers.{n}.layer.weight == tensor(3, channels[-n-1], 1, 1) random
    # - to_data_layers.{n}.layer.dense.bias == tensor(channels[-n-1]) with 1
    # - to_data_layers.{n}.layer.dense.layer.weight == tensor(channels[-n-1], 512) random
    std = 0.5
    g_state_dict = g_model['G_synthesis']['state_dict']
    g_new_state_dict = copy.deepcopy(g_state_dict)
    for e, channel in enumerate(channels[:(len(channels) - len(old_channels))][::-1]):

        idx = len(old_channels) + e

        # conv_blocks parameters
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.0.bias'] = torch.zeros(channels[-idx-1])
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.1.bias'] = torch.zeros(channels[-idx-1])
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.0.layer.weight'] = torch.zeros(1)
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.1.layer.weight'] = torch.zeros(1)
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.0.layer.layer.dense.layer.weight'] = torch.zeros(channels[-idx], 512).normal_(0, std)
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.1.layer.layer.dense.layer.weight'] = torch.zeros(channels[-idx-1], 512).normal_(0, std)
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.0.layer.layer.weight'] = torch.zeros(channels[-idx-1], channels[-idx], 3, 3).normal_(0, std)
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.1.layer.layer.weight'] = torch.zeros(channels[-idx-1], channels[-idx-1], 3, 3).normal_(0, std)
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.0.layer.layer.dense.bias'] = torch.ones(channels[-idx])
        g_new_state_dict[f'conv_blocks.{idx}.conv_block.1.layer.layer.dense.bias'] = torch.ones(channels[-idx-1])

        g_new_state_dict[f'conv_blocks.{idx}.conv_block.0.layer.layer.filter.filter_kernel'] = g_state_dict['conv_blocks.1.conv_block.0.layer.layer.filter.filter_kernel']

        # to_data parameters
        g_new_state_dict[f'to_data_layers.{idx}.bias'] = torch.ones(3)
        g_new_state_dict[f'to_data_layers.{idx}.layer.weight'] = torch.zeros(3, channels[-idx-1], 1, 1).normal_(0, std)
        g_new_state_dict[f'to_data_layers.{idx}.layer.dense.bias'] = torch.ones(channels[-idx-1])
        g_new_state_dict[f'to_data_layers.{idx}.layer.dense.layer.weight'] = torch.zeros(channels[-idx-1], 512).normal_(0, std)


    # Parameters in discriminator need to be modified:
    # - from_data_layers.0.bias == tensor(channels[0]) random
    # - from_data_layers.0.layer.weight == tensor(channels[0], 3, 1, 1) random
    # - shift all conv_blocks.{n}* to all_conv_blocks.{n + nb_new_channels}*
    # For the added channels:
    # - conv_blocks.0.conv_block.0.bias == tensor(channels[n])
    # - conv_blocks.0.conv_block.0.layer.weight == tensor(channels[n], channels[n], 3, 3)
    # - conv_blocks.0.conv_block.1.bias == tensor(channels[n+1])
    # - conv_blocks.0.conv_block.1.layer.weight == tensor(channels[n+1], channels[n], 3, 3])
    # - conv_blocks.0.conv_block.1.layer.filter.filter_kernel == tensor(1, 1, 4, 4])
    # - conv_blocks.0.projection.weight == tensor(channels[n+1], channels[n], 1, 1)
    # - conv_blocks.0.projection.filter.filter_kernel == tensor(1, 1, 4, 4)
    std = 0.5
    d_state_dict = d_model['state_dict']
    d_new_state_dict = copy.deepcopy(d_state_dict)
    d_new_state_dict['from_data_layers.0.bias'] = torch.zeros(channels[0]).normal_(0, std)
    d_new_state_dict['from_data_layers.0.layer.weight'] = torch.zeros(channels[0], 3, 1, 1).normal_(0, std)
    nb_shift = len(channels) - len(old_channels)
    keys_to_rename = {}
    for k in d_state_dict.keys():
        if 'conv_blocks' in k:
            i = k[12]
            new_i = str(int(i) + nb_shift)
            keys_to_rename[k] = k.replace(f'conv_blocks.{i}', f'conv_blocks.{new_i}')
            del d_new_state_dict[k]

    for k, k_ in sorted(keys_to_rename.items()):
        d_new_state_dict[k_] = d_state_dict[k]

    for e, channel in enumerate(channels[:(len(channels) - len(old_channels))]):
        d_new_state_dict[f'conv_blocks.{e}.conv_block.0.bias'] = torch.zeros(channel).normal_(0, std)
        d_new_state_dict[f'conv_blocks.{e}.conv_block.0.layer.weight'] = torch.zeros(channel, channel, 3, 3).normal_(0, std)
        d_new_state_dict[f'conv_blocks.{e}.conv_block.1.bias'] = torch.zeros(channels[e + 1]).normal_(0, std)
        d_new_state_dict[f'conv_blocks.{e}.conv_block.1.layer.weight'] = torch.zeros(channels[e+1], channels[e], 3, 3).normal_(0, std)
        d_new_state_dict[f'conv_blocks.{e}.conv_block.1.layer.filter.filter_kernel'] = d_state_dict['conv_blocks.0.conv_block.1.layer.filter.filter_kernel']
        d_new_state_dict[f'conv_blocks.{e}.projection.weight'] = torch.zeros(channels[e+1], channels[e], 1, 1).normal_(0, std)
        d_new_state_dict[f'conv_blocks.{e}.projection.filter.filter_kernel'] = d_state_dict['conv_blocks.0.projection.filter.filter_kernel']

    # Set new state dict
    g_model['G_synthesis']['state_dict'] = g_new_state_dict
    d_model['state_dict'] = d_new_state_dict

    return g_model, d_model



#########
# Run
#########

if __name__=='__main__':

    import matplotlib.pyplot as plt
    import json

    # Path to the model
    start_model_path = 'outputs/start_model/Gs.pt'
    model_path = 'outputs/checkpoints/11000_2020-10-12_09-28-30/Gs.pth'

    # Generate images
    #psi = 0.8
    #output_dir = 'outputs/generated_psi_08'
    #run_generate(model_path, output_dir, seeds=[50 + i for i in range(16)], truncation_psi=psi)
    #build_image([f'{output_dir}/{i}' for i in os.listdir(output_dir)], 'outputs/generated.png', nb_rows=4)

    # Interpolate
    #output_dir = 'outputs/interpolate'
    #run_interpolate(model_path, output_dir, 24, seeds=[3000, 4000], truncation_psi=0.5)
    #build_image(sorted([f'{output_dir}/{i}' for i in os.listdir(output_dir)]), 'outputs/interpolate.png', nb_rows = 5)

    # Projection
    #output_dir = 'outputs/projection'
    #run_projection(model_path, output_dir, 'inputs/projection', num_steps=1000, num_snapshots=5, gpu=True)

    # Evaluate a model
    #output_dir = 'outputs/evaluation'
    #model_learning(start_model_path, 'outputs/checkpoints', output_dir, [50 + i for i in range(10)])

    # Compute the metrics
    #output_dir = 'outputs/metrics'
    #compute_metrics(start_model_path, 'outputs/checkpoints', 'outputs/metrics', 'inputs/resized', 10000, sampling=1, gpu=True)

    #fids_10, fids_20, fids_50 = [], [], []
    #for i in sorted(os.listdir(output_dir), key=lambda x: int(x)):
    #    metrics = json.load(open(f'{output_dir}/{i}/metrics.json'))
    #    if 'FID:10k' in metrics:
    #        fids_10.append(metrics['FID:10k'])
    #    if 'FID:20k' in metrics:
    #        fids_20.append(metrics['FID:20k'])
    #    if 'FID:50k' in metrics:
    #        fids_50.append(metrics['FID:50k'])

    #plt.plot(fids_10, label='10') ; plt.plot(fids_20, label='20') ; plt.plot(fids_50, label='50')
    #plt.legend() ; plt.grid() ; plt.show()

    # Build higher-resolution model
    channels = [64, 128, 256, 512, 512, 512, 512, 512]
    g_model_path = 'outputs/start_model/Gs.pt'
    d_model_path = 'outputs/start_model/D.pt'
    g_model, d_model = build_model(g_model_path, d_model_path, channels)
    torch.save(g_model, 'outputs/start_model/Gs_512.pt')
    torch.save(d_model, 'outputs/start_model/D_512.pt')
