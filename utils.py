
import os
from subprocess import Popen
import shlex
from PIL import Image


DEFAULT_TRUNCATION_PSI = 0.5

#########
# Run commands
#########

def run_generate(model_path, out_path, seeds, truncation_psi=DEFAULT_TRUNCATION_PSI):
    """
    with higher psi, you can get higher diversity on the generated images but it also has a higher chance of generating weird or broken faces
    """

    seeds_str = ','.join(map(str, seeds))

    os.makedirs(out_path, exist_ok=True)
    cmd = f'python run_generator.py generate_images --network={model_path} --seeds={seeds_str} --output={out_path} --truncation_psi={truncation_psi}'
    Popen(shlex.split(cmd)).wait()


def run_interpolate(model_path, out_path, seeds, number, truncation_psi=DEFAULT_TRUNCATION_PSI):

    seeds_str = ','.join(map(str, seeds))

    os.makedirs(out_path, exist_ok=True)
    cmd = f'python run_generator.py interpolate --network={model_path} --seeds={seeds_str} --output={out_path} --number={number} --truncation_psi={truncation_psi}'
    Popen(shlex.split(cmd)).wait()


def run_metrics(model_path, out_path, data_dir, num_samples, gpu=True):
    
    os.makedirs(out_path, exist_ok=True)
    gpu_str = '--gpu=0' if gpu else ''
    cmd = f'python run_metrics.py fid --network={model_path} --num_samples={num_samples} --output={out_path} --data_dir={data_dir} {gpu_str}'
    Popen(shlex.split(cmd)).wait()


def run_projection(model_path, out_path, input_path, num_steps, num_snapshots=5, gpu=True):

    os.makedirs(out_path, exist_ok=True)
    gpu_str = '--gpu=0' if gpu else ''
    cmd = f'python run_projector.py project_real_images --network={model_path} --num_steps={num_steps} --num_snapshots={num_snapshots} --output={out_path} --data_dir={data_dir} {gpu_str}'
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


def compute_metrics(start_model_path, checkpoints_path, output_path, data_dir, num_samples, sampling=1, gpu=True):

    # List all models being evaluated
    checkpoints = sorted(os.listdir(checkpoints_path), key=lambda x: int(x.split('_')[0]))
    checkpoints = [c for e, c in enumerate(checkpoints) if (e + 1) % sampling == 0]
    models = [start_model_path] + [f'{checkpoints_path}/{c}/Gs.pth' for c in checkpoints]

    # Run the metrics
    for e, m in enumerate(models):
        print(f'Evaluating model {m}')
        run_metrics(m, f'{output_path}/{e}', data_dir, num_samples, gpu=True)




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
    #run_generate(model_path, output_dir, [50 + i for i in range(16)], truncation_psi=psi)
    #build_image([f'{output_dir}/{i}' for i in os.listdir(output_dir)], 'outputs/generated.png', nb_rows=4)

    # Interpolate
    #output_dir = 'outputs/interpolate'
    #run_interpolate(model_path, output_dir, [123, 5432], 24, truncation_psi=0.6)
    #build_image(sorted([f'{output_dir}/{i}' for i in os.listdir(output_dir)]), 'outputs/interpolate.png', nb_rows = 5)

    # Evaluate a model
    #output_dir = 'outputs/evaluation'
    #model_learning(start_model_path, 'outputs/checkpoints', output_dir, [50 + i for i in range(10)])

    # Compute the metrics
    #output_dir = 'outputs/metrics'
    #compute_metrics(start_model_path, 'outputs/checkpoints', 'outputs/metrics', 'inputs/resized', 10000, sampling=1, gpu=True)

    fids = []
    for i in sorted(os.listdir(output_dir), key=lambda x: int(x)):
        fids.append(json.load(open(f'{output_dir}/{i}/metrics.json'))['FID:1k'])

