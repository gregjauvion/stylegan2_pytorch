
from utils import run_generate, run_interpolate
from fast_srgan import get_model, srgan, srgan_dir

import cv2
import os


output_dir = 'outputs/jfr_gi_religion'
model = 'models/Gs_jfr_gi_religion.pth'
sr_model = get_model('models/fast_srgan_generator.h5')

seeds = list(range(1, 30))
nb_interpolation = 30

for psi in [0.25, 0.5, 0.75]:

    # Generated and interpolated images
    run_generate(model, f'{output_dir}/generation/{psi}/256', seeds, truncation_psi=psi)
    run_interpolate(model, f'{output_dir}/interpolation/{psi}/256', nb_interpolation, seeds=[6, 12], truncation_psi=psi, type_='slerp')

    # SR images
    srgan_dir(sr_model, f'{output_dir}/generation/{psi}/256', f'{output_dir}/generation/{psi}/1024')
    srgan_dir(sr_model, f'{output_dir}/interpolation/{psi}/256', f'{output_dir}/interpolation/{psi}/1024')
