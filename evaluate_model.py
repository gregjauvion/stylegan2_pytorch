
import os
from subprocess import Popen
import shlex


START_MODEL_PATH = 'outputs/start_model/Gs.pt'
ROOT = 'outputs/checkpoints'
OUT = 'outputs/images'


def run(model_path, out_path, number):

    os.makedirs(out_path, exist_ok=True)
    cmd = f'python run_generator.py generate_images --network={model_path} --seeds=5001-{5000 + number} --truncation_psi=0.5 --output={out_path}'
    Popen(shlex.split(cmd)).wait()


# Run start model
run(START_MODEL_PATH, f'{OUT}/0', 10)


# List the available checkpoints
checkpoints = os.listdir(ROOT)
for e, c in enumerate(checkpoints):
    model_path = f'{ROOT}/{c}/Gs.pth'
    run(model_path, f'{OUT}/{e + 1}', 10)
