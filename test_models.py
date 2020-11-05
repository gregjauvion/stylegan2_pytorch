
from utils import run_generate, run_interpolate
import cv2
from ISR.models import RDN, RRDN
import os



MODEL_CHURCH = 'models/Gs_church.pt'
MODEL_ARC = 'models/Gs_arc_dataset.pt'
MODEL_JFR = 'models/Gs_jfr_dataset.pt'
MODEL_JFR_RELIGION = 'models/Gs_jfr_dataset_religion.pt'
MODEL_JFR_GI_RELIGION = 'models/Gs_jfr_google_images_religion.pth'
MODEL_JFR_PARIS = 'models/Gs_jfr_dataset_paris.pt'


seeds = list(range(1, 17))

run_generate(MODEL_CHURCH, 'outputs/church/generated', seeds, truncation_psi=0.5)
run_generate(MODEL_ARC, 'outputs/arc/generated', seeds, truncation_psi=0.5)
run_generate(MODEL_JFR, 'outputs/jfr/generated', seeds, truncation_psi=0.5)
run_generate(MODEL_JFR_GI_RELIGION, 'outputs/jfr_gi_religion/generated', seeds, truncation_psi=0.5)
run_generate(MODEL_JFR_PARIS, 'outputs/jfr_paris/generated', seeds, truncation_psi=0.5)


# Enhance images resolution
sr_models = {
    'rdn_small': RDN(weights='psnr-small'),
    'rdn_large': RDN(weights='psnr-large'),
    'rdn_nc': RDN(weights='noise-cancel')
}

for model in ['jfr_gi_religion']:
    #for sr_model in ['rdn_small', 'rdn_large', 'rdn_nc']:
    for sr_model in ['rdn_nc']:#['rdn_large', 'rdn_nc']:
        print(model, sr_model)
        # Create output dir if needed
        in_path, out_path = f'outputs/{model}/generated', f'outputs/{model}/generated_{sr_model}_512'
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        for p in os.listdir(in_path):
            print(p)
            img = cv2.imread(f'{in_path}/{p}')
            #img_sr = sr_models[sr_model].predict(sr_models[sr_model].predict(img))
            img_sr = sr_models[sr_model].predict(img)
            cv2.imwrite(f'{out_path}/{p}', img_sr)
