
import os
from subprocess import Popen
import shlex
from PIL import Image


START_MODEL_PATH = 'outputs/start_model/Gs.pt'
ROOT = 'outputs/checkpoints'
OUT = 'outputs/images'


def run(model_path, out_path, number):

    os.makedirs(out_path, exist_ok=True)
    cmd = f'python run_generator.py generate_images --network={model_path} --seeds=5001-{5000 + number} --truncation_psi=0.5 --output={out_path}'
    Popen(shlex.split(cmd)).wait()


def evaluate():

    # Run start model
    run(START_MODEL_PATH, f'{OUT}/0', 10)

    # List the available checkpoints
    checkpoints = sorted(os.listdir(ROOT), key=lambda x: int(x.split('_')[0]))
    for e, c in enumerate(checkpoints):
        model_path = f'{ROOT}/{c}/Gs.pth'
        run(model_path, f'{OUT}/{e + 1}', 10)


def make_image():

    images_names = os.listdir(f'{OUT}/0')
    folders = list(map(str, sorted(map(int, filter(lambda x: x!='.DS_Store', os.listdir(OUT))))))

    # Build big image
    width, height = Image.open(f'{OUT}/{folders[0]}/{images_names[0]}').size
    total_width, total_height = width * len(folders), height * len(images_names)
    new_im = Image.new('RGB', (total_width, total_height))

    for e_f, folder in enumerate(folders):
        for e_n, name in enumerate(images_names):
            image = Image.open(f'{OUT}/{folder}/{name}')
            new_im.paste(image, (width * e_f, height * e_n))

    new_im.save(f'outputs/images/all.png')


def interpolate_image(input_dir, output_dir, nb_rows=1):

    images_names = os.listdir(input_dir)
    images_names = sorted(images_names, key=lambda x: int(x.replace('.png', '').split('_')[-1]))

    # Build image with interpolation
    width, height = Image.open(f'{input_dir}/{images_names[0]}').size
    nb_cols = 1 + (len(images_names) - 1) // nb_rows
    new_im = Image.new('RGB', (width * nb_cols, height * nb_rows))

    for e, i in enumerate(images_names):
        image = Image.open(f'{input_dir}/{i}')
        row, col = e // nb_cols, e - nb_cols * (e // nb_cols)
        new_im.paste(image, (col * width, row * height))

    new_im.save(f'{output_dir}/interpolate.png')


if __name__=='__main__':

    #evaluate()
    #make_image()
    interpolate_image('outputs/interpolate', 'outputs', nb_rows=3)
