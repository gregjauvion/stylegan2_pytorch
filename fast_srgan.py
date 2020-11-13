
from tensorflow import keras
import numpy as np
import cv2
import os
import tqdm


def get_model(path):

    model = keras.models.load_model(path)
    inputs = keras.Input((None, None, 3))
    output = model(inputs)

    return keras.models.Model(inputs, output)


def srgan(model, image_path, output_path):

    # Read image
    low_res = cv2.imread(image_path, 1)

    # Convert to RGB (opencv uses BGR as default)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

    # Rescale to 0-1.
    low_res = low_res / 255.0

    # Get super resolution image
    sr = model.predict(np.expand_dims(low_res, axis=0))[0]

    # Rescale values in range 0-255
    sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

    # Convert back to BGR for opencv
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

    # Save the results:
    cv2.imwrite(output_path, sr)


def srgan_dir(model, input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    for img in tqdm.tqdm(os.listdir(input_dir)):
        srgan(model, f'{input_dir}/{img}', f'{output_dir}/{img}')



if __name__ == '__main__':
    
    model = get_model('models/fast_srgan_generator.h5')
    srgan(model, 'outputs/arc/generated/seed_0001.png', 'outputs/arc/test.png')
