import tensorflow_hub as hub
import numpy as np
import os

from utils import load_image_into_numpy_array, COCO17_HUMAN_POSE_KEYPOINTS, path2model, path2config, category_index

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

import collections
import six
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2

from object_detection.utils import config_util
from object_detection.builders import model_builder

class ImageInference:
    def __init__(self):

        image_path = "/Users/dharrensandhi/fiftyone/coco-2017/validation/data/000000041990.jpg"
        video_path = "/Users/dharrensandhi/Downloads/000017.mp4"
        self.STANDARD_COLORS = [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen'
        ]
        self.box_num = 1
        self.num_detections = 0

        self.keypoint_coordinates = {}
        self.model = self.load_model()
        self.image = self.image_selection(image_path)
        self.result = self.inference()
        # self.visualisation_image()
        # self.visualisation_video_live()
        self.visualisation_video_offline(video_path)

    def load_model(self):
        # print('loading model...')
        # hub_model = hub.load(model_handle)
        # print('model loaded!')
        # configs = config_util.get_configs_from_pipeline_file(path2config)  # importing config
        # model_config = configs['model']  # recreating model config
        # detection_model = model_builder.build(model_config=model_config, is_training=False)  # importing model
        # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        # ckpt.restore(os.path.join(f'{path2model}/checkpoint', 'ckpt-0')).expect_partial()

        detection_model = tf.saved_model.load(f'/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/model_1_scripts/saved_model/saved_model')

        return detection_model

    def detect_fn(self, image):
        """
        Detect objects in image.

        Args:
          image: (tf.tensor): 4D input image

        Returs:
          detections (dict): predictions that model made
        """

        image, shapes = self.model.preprocess(image)
        prediction_dict = self.model.predict(image, shapes)
        detections = self.model.postprocess(prediction_dict, shapes)

        return detections

    def image_selection(self, image_path):

        flip_image_horizontally = False
        convert_image_to_grayscale = False

        image_np = load_image_into_numpy_array(image_path)

        # Flip horizontally
        if flip_image_horizontally:
            image_np[0] = np.fliplr(image_np[0]).copy()

        # Convert image to grayscale
        if convert_image_to_grayscale:
            image_np[0] = np.tile(np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        return image_np

    def inference(self):
        # running inference
        input_tensor = tf.convert_to_tensor(self.image, dtype=tf.uint8)
        # results = self.detect_fn(input_tensor)
        infer = self.model.signatures["serving_default"]
        results = infer(input_tensor=input_tensor)

        # different object detection models have additional results
        # all of them are explained in the documentation
        result = {key: value.numpy() for key, value in results.items()}

        print(result.keys())

        return result

    def visualisation_image(self):
        label_id_offset = 1
        image_np_with_detections = self.image.copy()

        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in self.result:
            keypoints = self.result['detection_keypoints'][0]
            keypoint_scores = self.result['detection_keypoint_scores'][0]

        self.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections[0],
            self.result['detection_boxes'][0],
            (self.result['detection_classes'][0] + label_id_offset).astype(int),
            self.result['detection_scores'][0],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

        print(self.keypoint_coordinates)

        plt.figure(figsize=(24, 32))
        plt.imshow(image_np_with_detections[0])
        # plt.show()
        plt.savefig("/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/model_1_scripts/output_images/inference.png")


    def visualisation_video_live(self):
        cap = cv2.VideoCapture(0)

        while True:
            # Read frame from camera
            ret, image_np = cap.read()

            resized_frame = cv2.resize(image_np, (640, 480))

            input_tensor = tf.convert_to_tensor(np.expand_dims(resized_frame, 0), dtype=tf.float32)
            results = self.detect_fn(input_tensor)

            result = {key: value.numpy() for key, value in results.items()}

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # Use keypoints if available in detections
            keypoints, keypoint_scores = None, None
            if 'detection_keypoints' in result:
                keypoints = result['detection_keypoints'][0]
                keypoint_scores = result['detection_keypoint_scores'][0]

            self.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                result['detection_boxes'][0],
                (result['detection_classes'][0] + label_id_offset).astype(int),
                result['detection_scores'][0],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def visualisation_video_offline(self, path):

        vidObj = cv2.VideoCapture(path)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # video compression format
        HEIGHT = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))  # webcam video frame height
        WIDTH = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))  # webcam video frame width
        FPS = int(vidObj.get(cv2.CAP_PROP_FPS))  # webcam video frame rate

        video_name = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/model_1_scripts/keypoint_video/keypoint_video_test.avi"
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))

        count = 0

        while True:
            # Read frame from camera
            success, image = vidObj.read()

            resized_frame = cv2.resize(image, (512, 512))

            input_tensor = tf.convert_to_tensor(np.expand_dims(resized_frame, 0), dtype=tf.uint8)
            # results = self.detect_fn(input_tensor)
            infer = self.model.signatures["serving_default"]
            results = infer(input_tensor=input_tensor)

            result = {key: value.numpy() for key, value in results.items()}

            label_id_offset = 1
            image_np_with_detections = image.copy()

            # Use keypoints if available in detections
            keypoints, keypoint_scores = None, None
            if 'detection_keypoints' in result:
                keypoints = result['detection_keypoints'][0]
                keypoint_scores = result['detection_keypoint_scores'][0]

            self.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                result['detection_boxes'][0],
                (result['detection_classes'][0] + label_id_offset).astype(int),
                result['detection_scores'][0],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

            # Write to the video file
            print(f"Frame {count} completed")
            count += 1

            if count == 50:
                break

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()


    def visualize_boxes_and_labels_on_image_array(self,
            image,
            boxes,
            classes,
            scores,
            category_index,
            keypoints=None,
            keypoint_scores=None,
            keypoint_edges=None,
            track_ids=None,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=.5,
            agnostic_mode=False,
            line_thickness=4,
            groundtruth_box_visualization_color='black',
            skip_scores=False,
            skip_labels=False,
            skip_track_ids=False):
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_keypoints_map = collections.defaultdict(list)
        box_to_keypoint_scores_map = collections.defaultdict(list)
        box_to_track_ids_map = {}
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(boxes.shape[0]):
            if max_boxes_to_draw == len(box_to_color_map):
                break
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if keypoint_scores is not None:
                    box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
                if track_ids is not None:
                    box_to_track_ids_map[box] = track_ids[i]
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ''
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in six.viewkeys(category_index):
                                class_name = category_index[classes[i]]['name']
                            else:
                                class_name = 'N/A'
                            display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = '{}%'.format(round(100 * scores[i]))
                        else:
                            display_str = '{}: {}%'.format(display_str, round(100 * scores[i]))
                    if not skip_track_ids and track_ids is not None:
                        if not display_str:
                            display_str = 'ID {}'.format(track_ids[i])
                        else:
                            display_str = '{}: ID {}'.format(display_str, track_ids[i])
                    box_to_display_str_map[box].append(display_str)
                    if agnostic_mode:
                        box_to_color_map[box] = 'DarkOrange'
                    elif track_ids is not None:
                        prime_multipler = self._get_multiplier_for_color_randomness()
                        box_to_color_map[box] = self.STANDARD_COLORS[
                            (prime_multipler * track_ids[i]) % len(self.STANDARD_COLORS)]
                    else:
                        box_to_color_map[box] = self.STANDARD_COLORS[
                            classes[i] % len(self.STANDARD_COLORS)]

        # Draw all boxes onto image.
        for box, color in box_to_color_map.items():
            self.keypoint_coordinates[f'Box {self.box_num}'] = []
            if keypoints is not None:
                keypoint_scores_for_box = None
                if box_to_keypoint_scores_map:
                    keypoint_scores_for_box = box_to_keypoint_scores_map[box]
                self.draw_keypoints_on_image_array(
                    image,
                    box_to_keypoints_map[box],
                    keypoint_scores_for_box,
                    min_score_thresh=min_score_thresh,
                    color=color,
                    radius=line_thickness / 2,
                    use_normalized_coordinates=use_normalized_coordinates,
                    keypoint_edges=keypoint_edges,
                    keypoint_edge_color=color,
                    keypoint_edge_width=line_thickness // 2)

            self.box_num += 1

        return image

    def _get_multiplier_for_color_randomness(self):
        """Returns a multiplier to get semi-random colors from successive indices.

        This function computes a prime number, p, in the range [2, 17] that:
        - is closest to len(STANDARD_COLORS) / 10
        - does not divide len(STANDARD_COLORS)

        If no prime numbers in that range satisfy the constraints, p is returned as 1.

        Once p is established, it can be used as a multiplier to select
        non-consecutive colors from STANDARD_COLORS:
        colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
        """
        num_colors = len(self.STANDARD_COLORS)
        prime_candidates = [5, 7, 11, 13, 17]

        # Remove all prime candidates that divide the number of colors.
        prime_candidates = [p for p in prime_candidates if num_colors % p]
        if not prime_candidates:
            return 1

        # Return the closest prime number to num_colors / 10.
        abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
        num_candidates = len(abs_distance)
        inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
        return prime_candidates[inds[0]]

    def draw_keypoints_on_image_array(self, image,
                                      keypoints,
                                      keypoint_scores=None,
                                      min_score_thresh=0.5,
                                      color='red',
                                      radius=2,
                                      use_normalized_coordinates=True,
                                      keypoint_edges=None,
                                      keypoint_edge_color='green',
                                      keypoint_edge_width=2):

        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        self.draw_keypoints_on_image(image_pil,
                                keypoints,
                                keypoint_scores=keypoint_scores,
                                min_score_thresh=min_score_thresh,
                                color=color,
                                radius=radius,
                                use_normalized_coordinates=use_normalized_coordinates,
                                keypoint_edges=keypoint_edges,
                                keypoint_edge_color=keypoint_edge_color,
                                keypoint_edge_width=keypoint_edge_width)
        np.copyto(image, np.array(image_pil))

    def draw_keypoints_on_image(self, image,
                                keypoints,
                                keypoint_scores=None,
                                min_score_thresh=0.5,
                                color='red',
                                radius=2,
                                use_normalized_coordinates=True,
                                keypoint_edges=None,
                                keypoint_edge_color='green',
                                keypoint_edge_width=2):
        """Draws keypoints on an image.

        Args:
          image: a PIL.Image object.
          keypoints: a numpy array with shape [num_keypoints, 2].
          keypoint_scores: a numpy array with shape [num_keypoints].
          min_score_thresh: a score threshold for visualizing keypoints. Only used if
            keypoint_scores is provided.
          color: color to draw the keypoints with. Default is red.
          radius: keypoint radius. Default value is 2.
          use_normalized_coordinates: if True (default), treat keypoint values as
            relative to the image.  Otherwise treat them as absolute.
          keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.
          keypoint_edge_color: color to draw the keypoint edges with. Default is red.
          keypoint_edge_width: width of the edges drawn between keypoints. Default
            value is 2.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        keypoints = np.array(keypoints)
        keypoints_x = [k[1] for k in keypoints]
        keypoints_y = [k[0] for k in keypoints]
        if use_normalized_coordinates:
            keypoints_x = tuple([im_width * x for x in keypoints_x])
            keypoints_y = tuple([im_height * y for y in keypoints_y])
        if keypoint_scores is not None:
            keypoint_scores = np.array(keypoint_scores)
            valid_kpt = np.greater(keypoint_scores, min_score_thresh)
        else:
            valid_kpt = np.where(np.any(np.isnan(keypoints), axis=1),
                                 np.zeros_like(keypoints[:, 0]),
                                 np.ones_like(keypoints[:, 0]))
        valid_kpt = [v for v in valid_kpt]

        for keypoint_x, keypoint_y, valid in zip(keypoints_x, keypoints_y, valid_kpt):
            if valid:
                coordinate = [keypoint_x, keypoint_y]
                self.keypoint_coordinates[f'Box {self.box_num}'].append(coordinate)
                draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                              (keypoint_x + radius, keypoint_y + radius)],
                             outline=color, fill=color)
        if keypoint_edges is not None:
            for keypoint_start, keypoint_end in keypoint_edges:
                if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
                        keypoint_end < 0 or keypoint_end >= len(keypoints)):
                    continue
                if not (valid_kpt[keypoint_start] and valid_kpt[keypoint_end]):
                    continue
                edge_coordinates = [
                    keypoints_x[keypoint_start], keypoints_y[keypoint_start],
                    keypoints_x[keypoint_end], keypoints_y[keypoint_end]
                ]
                draw.line(
                    edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)



if __name__ == '__main__':
    imageInference = ImageInference()

# python /Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/models/research/object_detection/exporter_main_v2.py \
# --input_type image_tensor \
# --pipeline_config_path "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/pipeline_coco.config" \
# --trained_checkpoint_dir "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/output_models" \
# --output_directory "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/saved_model"