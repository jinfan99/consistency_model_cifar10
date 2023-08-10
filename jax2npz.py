import torch
import jax
import io
import os.path as osp
from glob import glob
import numpy as np
import os

INPUT_DIR = '/home/ubuntu/tmp_images/sscd'

def main():
    pth_files = glob(osp.join(INPUT_DIR, '*.pth'))

    for image_file in pth_files:
        images = torch.load(image_file)
        model_name = osp.basename(image_file).split('.')[0]
        output_dir = osp.join(INPUT_DIR, model_name)

        print('processing ', model_name, '......')

        if not osp.exists(output_dir):
            os.mkdir(output_dir)

        bs = 1000
        for r in range(10):
            samples = np.array(images[r*bs:(r+1)*bs]).astype(np.uint8)
            with open(
            os.path.join(output_dir, f"samples_{r}.npz"), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, samples=samples)
                fout.write(io_buffer.getvalue())

if __name__ == '__main__':
    main()