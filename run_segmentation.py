import sys
import os
import glob
import imageio
import numpy as np
from maskrcnn_utils import InferenceConfig
from maskrcnn_utils import Dataset
from mrcnn import model as modellib


def main():
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    model_dir = '.'
    overlap = 64
    files = glob.glob(os.path.join(inpath,'*-B.tif'))
    dataset = Dataset()
    dataset.load_files(files)
    dataset.prepare()
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode = "inference",
                              config = inference_config,
                              model_dir = model_dir)
    model.load_weights(os.path.join(model_dir,'weights.h5'), by_name=True)

    for i,image_id in enumerate(dataset.image_ids):
        tiles = dataset.load_image(image_id, tile_overlap=overlap)
        orig_size = dataset.get_orig_size(image_id)
        mask_img = np.zeros(orig_size, dtype=np.uint8)

        tile_masks = []
        for image in tiles:
            mask = model.detect([image], verbose=0)[0]
            tile_masks.append(mask)

        mask_img = dataset.merge_tiles(image_id, tile_masks)
        imageio.imwrite(os.path.join(outpath,os.path.basename(files[i])), mask_img)


if __name__ == "__main__":
    main()
