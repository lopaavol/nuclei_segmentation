# nuclei_segmentation
Python utilities to apply pre-trained Mask R-CNN model for nuclei segmentation.

The Mask R-CNN implementation is based on Matterport Mask R-CNN v2.1:\
https://github.com/matterport/Mask_RCNN \
The code is included in the repository separately for TensorFlow 1.x and TensorFlow 2.x so there is no need to separately install Mask R-CNN library. The TensorFlow 2.x modified version of Mask R-CNN was made by leekunhee:\
https://github.com/leekunhee/Mask_RCNN

Requirements:
- Download the pre-trained weights into the root directory of the repository: https://drive.google.com/uc?id=19EmZ57LXSArG-Z1HC8NOtrQUxGNzW3vv
- The version of Mask R-CNN requires Keras and TensorFlow 1.x. It is working at least with keras 2.2.4 and likely newer versions. The repository includes different Mask R-CNN implementations for TensorFlow 1.x and 2.x. The implementation has been tested functional with TensorFlow 1.13.2 and TensorFlow 2.2.0.
- Some other typical Python libraries such as numpy, skimage, imageio,...

Installation:
Clone the repository, download the pre-trained weights, and rename either mrcnn_tf1 or mrcnn_tf2 to 'mrcnn' depending on the version of TensorFlow that is installed into your machine.
