import sys
import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from maskrcnn_utils import InferenceConfig
from maskrcnn_utils import Dataset
from mrcnn import model as modellib


def main():
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    model_dir = '.'
    overlap = 128
    filter_edge = True

    ipath = Path(inpath)
    opath = Path(outpath)
    files = ipath.rglob('*DAPI*.tif')
    files = [x for x in files]

    dataset = Dataset()
    dataset.load_files(files)
    dataset.prepare()

    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode = "inference",
                              config = inference_config,
                              model_dir = model_dir)
    model.load_weights(os.path.join(model_dir,'weights.h5'), by_name=True)

    for i,image_id in enumerate(tqdm(dataset.image_ids)):
        tiles = dataset.load_image(image_id, tile_overlap=overlap)
        orig_size = dataset.get_orig_size(image_id)
        mask_img = np.zeros(orig_size, dtype=np.uint8)

        tile_masks = []
        for image in tiles:
            mask = model.detect([image], verbose=0)[0]
            tile_masks.append(mask)

        mask_img = dataset.merge_tiles(image_id, tile_masks, filter_edge=filter_edge)
        suffix = files[i].suffix
        imageio.imwrite(opath / files[i].name.replace(suffix,'.png'), mask_img)


if __name__ == "__main__":
    main()
