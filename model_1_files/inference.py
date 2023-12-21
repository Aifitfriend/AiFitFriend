import numpy as np
import os

from .utils import load_image_into_numpy_array, COCO17_HUMAN_POSE_KEYPOINTS, path2model, path2config, category_index

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

import collections
import six
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2

from .models.research.object_detection.utils import config_util
from .models.research.object_detection.builders import model_builder

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class ImageInference():
    def __init__(self,video_filename):

        # image_path = "/Users/dharrensandhi/fiftyone/coco-2017/validation/data/000000041990.jpg"
        video_path = "model_1_files/upload_video/" + video_filename
        action_model_path = "model_2_files/lstm_attention_128HUs_dharren_2d_0.0001reg_100epoch_70-30-30_16batch_17kp_nopress.h5"

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

        self.curl_counter = 0
        self.curl_counter_wrong = 0
        self.squat_counter = 0
        self.curl_stage = None
        self.left_curl_stage = None
        self.right_curl_stage = None
        self.curl_stage_wrong = None
        self.curl_stage_wrong_refresh = None
        self.squat_stage = None
        self.left_rep_complete = False
        self.right_rep_complete = False
        self.curl_wrong = False

        self.sequence_length = 30
        self.predictions = []
        self.actions = ['curl', 'squat']  # Replace with your action names

        self.keypoint_coordinates = {}
        self.keypoint_coordinates_per_frame = []
        self.keypoint_coordinates_per_frame_flatten = []

        self.keypoint_model = self.load_keypoint_model(mode="load_from_tflite")
        self.action_model = self.load_action_recognition_model(action_model_path)

        self.image = None
        # self.image = self.image_selection(image_path)
        # self.result = self.inference()
        # self.visualisation_image(image_path=image_path)
        # self.visualisation_video_live()
        self.visualisation_video_offline(video_path)

    def load_keypoint_model(self, mode):
        print('Model Loading')
        # hub_model = hub.load(model_handle)
        # print('model loaded!')

        ## built from config
        if mode == "build_from_config":
            configs = config_util.get_configs_from_pipeline_file(path2config)  # importing config
            model_config = configs['model']  # recreating model config
            detection_model = model_builder.build(model_config=model_config, is_training=False)  # importing model
            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(os.path.join(f'{path2model}/checkpoint', 'ckpt-0')).expect_partial()

        ## load model directly from pb file
        elif mode == "load_from_pb":
            detection_model = tf.saved_model.load(f'/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/model_1_scripts/saved_model/saved_model')

        ## load model from tflite
        elif mode == "load_from_tflite":
            detection_model = tf.lite.Interpreter(model_path="saved_model/saved_model.tflite")
            detection_model.allocate_tensors()

        print("Model Loaded")

        return detection_model

    def load_action_recognition_model(self, model_path):

        action_model = load_model(model_path)

        return action_model

    def detect_fn(self, image):
        """
        Detect objects in image.

        Args:
          image: (tf.tensor): 4D input image

        Returs:
          detections (dict): predictions that model made
        """

        image, shapes = self.keypoint_model.preprocess(image)
        prediction_dict = self.keypoint_model.predict(image, shapes)
        detections = self.keypoint_model.postprocess(prediction_dict, shapes)

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

    def keypoint_inference(self, mode):
        ## running inference from building saved model from config
        if mode == "build_from_config":
            input_tensor = tf.convert_to_tensor(self.image, dtype=tf.float32)
            results = self.detect_fn(input_tensor)
            result = {key: value.numpy() for key, value in results.items()}

        ## running inference from pb file directly
        elif mode == "load_from_pb":
            input_tensor = tf.convert_to_tensor(self.image, dtype=tf.uint8)
            infer = self.keypoint_model.signatures["serving_default"]
            results = infer(input_tensor=input_tensor)
            result = {key: value.numpy() for key, value in results.items()}

        ## running inference from tflite model
        elif mode == "load_from_tflite":
            input_details = self.keypoint_model.get_input_details()

            input_tensor = tf.convert_to_tensor(self.image, dtype=tf.uint8)
            self.keypoint_model.resize_tensor_input(input_details[0]['index'], input_tensor.shape, strict=False)
            self.keypoint_model.allocate_tensors()

            self.keypoint_model.set_tensor(input_details[0]['index'], input_tensor)
            self.keypoint_model.invoke()

            output_details = self.keypoint_model.get_output_details()
            result = {detail['name']: self.keypoint_model.get_tensor(detail['index']) for detail in output_details}

        # print(result.keys())

        return result

    def visualisation_image(self, image_path):
        label_id_offset = 1

        self.image = self.image_selection(image_path)
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

        #video_name = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/model_1_scripts/keypoint_video/keypoint_video_test.avi"
        #out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
  
        count = 0

        while True:
            # Read frame from camera
            success, frame_image = vidObj.read()

            resized_frame = cv2.resize(frame_image, (512, 512))
            expand_dim_image = np.expand_dims(resized_frame, 0)

            self.image = expand_dim_image

            result_keypoint = self.keypoint_inference(mode="load_from_tflite")

            label_id_offset = 1
            image_np_with_detections = frame_image.copy()
            
            # Use keypoints if available in detections
            keypoints, keypoint_scores = None, None
            detection_keypoints = "StatefulPartitionedCall:4"
            detection_keypoint_scores = "StatefulPartitionedCall:3"
            detection_boxes = "StatefulPartitionedCall:0"
            detection_classes = "StatefulPartitionedCall:2"
            detection_scores = "StatefulPartitionedCall:6"
            if detection_keypoints in result_keypoint:
                keypoints = result_keypoint[detection_keypoints][0]
                keypoint_scores = result_keypoint[detection_keypoint_scores][0]

            self.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                result_keypoint[detection_boxes][0],
                (result_keypoint[detection_classes][0] + label_id_offset).astype(int),
                result_keypoint[detection_scores][0],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

            self.keypoint_coordinates_per_frame_flatten = self.keypoint_coordinates_per_frame_flatten[-self.sequence_length:]

            if len(self.keypoint_coordinates_per_frame_flatten) == self.sequence_length:
                result_action = self.action_model.predict(np.expand_dims(self.keypoint_coordinates_per_frame_flatten, axis=0), verbose=0)[0]
                self.predictions.append(np.argmax(result_action))
                current_action = self.actions[np.argmax(result_action)]
                confidence = np.max(result_action)

                if confidence > 0.5:
                    print(current_action)
                else:
                    current_action = ''
                    print(current_action)

                # Count reps
                try:
                    self.count_reps(frame_image, current_action)
                except:
                    pass

            # Write to the video file
            if success:
                #out.write(image_np_with_detections)
                print(f"Frame {count} completed")
                count += 1

            if count == 150:
                break

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        #out.release()
        cv2.destroyAllWindows()

    def calculate_angle(self, a, b, c):
        """
        Computes 2D joint angle inferred by 3 keypoints and their relative positions to one another

        """
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def is_unstable_shoulder(self, left_shoulder, left_elbow, right_shoulder, right_elbow):
        # Define a threshold for elbow alignment
        elbow_alignment_threshold = 10.0  # Adjust this value based on your specific criteria

        # Check the angle between the shoulders and elbows
        left_shoulder_angle = self.calculate_angle(left_shoulder, left_elbow, [left_elbow[0], left_elbow[1] - 10])
        right_shoulder_angle = self.calculate_angle(right_shoulder, right_elbow, [right_elbow[0], right_elbow[1] - 10])
        # left_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_hip)
        # right_shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_hip)

        # If the angle exceeds the threshold, consider elbows flaring
        return abs(left_shoulder_angle - right_shoulder_angle) > elbow_alignment_threshold

    def count_reps(self, image, current_action):
        """
        Counts repetitions of each exercise. Global count and stage (i.e., state) variables are updated within this function.

        """

        incorrect_frame_path = "output_images/incorrect_frame"

        if current_action == 'curl':
            # Get coords
            left_shoulder = self.keypoint_coordinates_per_frame[-1][5]
            left_elbow = self.keypoint_coordinates_per_frame[-1][7]
            left_wrist = self.keypoint_coordinates_per_frame[-1][9]
            left_hip = self.keypoint_coordinates_per_frame[-1][11]

            right_shoulder = self.keypoint_coordinates_per_frame[-1][6]
            right_elbow = self.keypoint_coordinates_per_frame[-1][8]
            right_wrist = self.keypoint_coordinates_per_frame[-1][10]
            right_hip = self.keypoint_coordinates_per_frame[-1][12]

            # calculate elbow angle
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Pose correction for shoulder stability
            if self.is_unstable_shoulder(left_shoulder, left_elbow, right_shoulder, right_elbow):
                # Provide feedback or take corrective action for flaring elbows
                print('Your elbows are flaring!')
                height, width, _ = image.shape

                text = 'Your elbows are flaring'
                font_scale = 1.0
                font_thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX

                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_position = ((width - text_size[0]) // 2, 50)  # Adjust 50 for vertical position

                cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 
                            font_thickness, cv2.LINE_AA)
                cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 
                            font_thickness +1,cv2.LINE_AA)
                cv2.imwrite(incorrect_frame_path+"unstable_"+str(self.curl_counter_wrong)+".jpg", image)
                self.curl_wrong = True

            if left_angle < 30:
                self.left_curl_stage = "up"

            if right_angle < 30:
                self.right_curl_stage = "up"

            if left_angle > 140 and self.left_curl_stage == "up":
                self.left_curl_stage = "down"
                self.left_rep_complete = True

            if right_angle > 140 and self.right_curl_stage == "up":
                self.right_curl_stage = "down"
                self.right_rep_complete = True

            if self.left_rep_complete and self.right_rep_complete and not self.curl_wrong:
                self.curl_counter += 1
                self.left_rep_complete = False
                self.right_rep_complete = False

            elif self.left_rep_complete and self.right_rep_complete and self.curl_wrong:
                self.curl_counter_wrong += 1
                self.curl_wrong = False
                self.left_rep_complete = False
                self.right_rep_complete = False

            # # curl counter logic
            # if left_angle < 30 and right_angle < 30:
            #     self.curl_stage = "up"
            #
            # if left_angle > 140 and right_angle > 140:
            #     if self.curl_stage == 'up':
            #         self.curl_stage = "down"
            #         self.curl_counter += 1
            #         self.curl_stage_wrong_refresh = 'up'
            #     if self.curl_stage_wrong == 'up':
            #         self.curl_stage_wrong = "down"
            #         self.curl_stage_wrong_refresh = 'down'
            #         self.curl_stage = 'down'
            #         self.curl_counter_wrong += 1

            print(f"Left Curl Stage: {self.left_curl_stage}")
            print(f"Right Curl Stage: {self.right_curl_stage}")
            print(f"Correct Curl Counter: {self.curl_counter}")
            print(f"Incorrect Curl Counter: {self.curl_counter_wrong}")

            self.squat_stage = None

            # # Viz joint angle
            # viz_joint_angle(image, left_angle, left_elbow)
            # viz_joint_angle(image, right_angle, right_elbow)

        elif current_action == 'squat':
            # Get coords
            # left side
            left_shoulder = self.keypoint_coordinates_per_frame[-1][5]
            left_hip = self.keypoint_coordinates_per_frame[-1][11]
            left_knee = self.keypoint_coordinates_per_frame[-1][13]
            left_ankle = self.keypoint_coordinates_per_frame[-1][15]
            # right side
            right_shoulder = self.keypoint_coordinates_per_frame[-1][6]
            right_hip = self.keypoint_coordinates_per_frame[-1][12]
            right_knee = self.keypoint_coordinates_per_frame[-1][14]
            right_ankle = self.keypoint_coordinates_per_frame[-1][16]

            # Calculate knee angles
            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            # Calculate hip angles
            left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)

            # Squat counter logic
            thr = 165
            if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (
                    right_hip_angle < thr):
                self.squat_stage = "down"
            if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (
                    right_hip_angle > thr) and (self.squat_stage == 'down'):
                self.squat_stage = 'up'
                self.squat_counter += 1
            self.curl_stage = None
            self.curl_stage_wrong = None

            # # Viz joint angles
            # viz_joint_angle(image, left_knee_angle, left_knee)
            # viz_joint_angle(image, left_hip_angle, left_hip)

        else:
            pass

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

        keypoints_with_vis = []
        for coordinate in keypoints:
            x_coor = coordinate[1]
            y_coor = coordinate[0]
            keypoints_with_vis.append([x_coor, y_coor, 1])

        keypoints_with_vis_normal = np.array(keypoints_with_vis)
        self.keypoint_coordinates_per_frame.append(keypoints_with_vis_normal)

        keypoints_with_vis_flatten = np.array(keypoints_with_vis).flatten()
        self.keypoint_coordinates_per_frame_flatten.append(keypoints_with_vis_flatten)

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

    def get_curl_counter(self):
        return self.curl_counter

    def get_curl_counter_wrong(self):
        return self.curl_counter_wrong

if __name__ == '__main__':
    imageInference = ImageInference()