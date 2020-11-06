# nuclei_segmentation
Python utilities to apply pre-trained Mask R-CNN model for nuclei segmentation.

The Mask R-CNN implementation is based on Matterport Mask R-CNN v2.1:\
https://github.com/matterport/Mask_RCNN \
The code is now included in mrcnn directory due to minor modifications so there is no need to separately install Matterport Mask R-CNN.

Requirements:
- Pre-trained weights: https://drive.google.com/uc?id=19EmZ57LXSArG-Z1HC8NOtrQUxGNzW3vv
- The version of Mask R-CNN requires keras and tensorflow 1.x. It is working at least with keras 2.3.1 and tensorflow 1.13.2
- Some other typical Python libraries such as numpy, skimage, imageio,...
