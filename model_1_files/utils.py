import numpy as np
from six import BytesIO
from PIL import Image
from six.moves.urllib.request import urlopen

from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import ops as utils_ops

import tensorflow as tf

tf.get_logger().setLevel('ERROR')


def load_image_into_numpy_array(path):
 """Load an image from file into a numpy array.

 Puts image into numpy array to feed into tensorflow graph.
 Note that by convention we put it into a numpy array with shape
 (height, width, channels), where channels=3 for RGB.

 Args:
   path: the file path to the image

 Returns:
   uint8 numpy array with shape (img_height, img_width, 3)
 """
 image = None
 if (path.startswith('http')):
  response = urlopen(path)
  image_data = response.read()
  image_data = BytesIO(image_data)
  image = Image.open(image_data)
 else:
  image_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(image_data))

 (im_width, im_height) = image.size
 return np.array(image.getdata()).reshape(
  (1, im_height, im_width, 3)).astype(np.uint8)

MODEL = {
'CenterNet HourGlass104 Keypoints 512x512' : 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1'
}

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]

# PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#
# model_display_name = 'CenterNet HourGlass104 Keypoints 512x512'
# model_handle = MODEL[model_display_name]

path2config ='/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/saved_model_2/pipeline.config'
path2model = '/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/saved_model_2/'
path2label_map = '/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)
