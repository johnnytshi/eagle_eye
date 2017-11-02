import numpy as np
import os
import sys
import tensorflow as tf
import argparse
import cv2
import multiprocessing
from multiprocessing import Queue, Pool
from collections import defaultdict
import time

sys.path.append('/models/research/object_detection/')
from utils import label_map_util
from utils import visualization_utils as vis_util
from slackclient import SlackClient

MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/model_zoo/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/models/research/object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def detect_objects(output_q, image_np, sess, image_tensor, boxes, scores, classes, num_detections, min_score_thresh):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    scores_squeezed = np.squeeze(scores)
    for val in scores_squeezed:
        if val >= min_score_thresh:
            break
    else:
        print("NO object detected")
        return

    print("Object detected")

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        scores_squeezed,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=min_score_thresh,
        line_thickness=8)
    output_q.put(image_np)

def input_worker(input_q, output_q, min_score_thresh):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    while True:
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_objects(output_q, frame_rgb, sess, image_tensor, boxes, scores, classes, num_detections, min_score_thresh)

    sess.close()

def output_worker(output_q, slack_token, slack_channel):
    sc = SlackClient(slack_token)
    while True:
        frame = output_q.get()
        cv2.imwrite("image.jpg", frame)
        in_file = open("image.jpg", "rb") # opening for [r]eading as [b]inary
        data = in_file.read()
        print("Uploading to slack")
        ret = sc.api_call("files.upload", filename="image.jpg", channels=slack_channel, file=data)
        print(ret)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        help='rtsp://admin:*@192.*.*.*:554/h264/ch1/main/av_stream')
    parser.add_argument('-token', '--slack_token', dest='slack_token', type=str,
                        help='Slack Web API Token')
    parser.add_argument('-channel', '--slack_channel', dest='slack_channel', type=str,
                        help='Slack Web API Token')
    parser.add_argument('-min_score_thresh', '--min_score_thresh', dest='min_score_thresh', type=float,
                        default=0.5, help='percentage as min confidence score')
    args = parser.parse_args()
    print(args)

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(24)
    output_q = Queue(24)
    input_pool = Pool(2, input_worker, (input_q, output_q, args.min_score_thresh))
    output_pool = Pool(1, output_worker, (output_q, args.slack_token, args.slack_channel))

    video_capture = cv2.VideoCapture(args.video_source)
    while(1):
        ret, frame = video_capture.read()

        if ret:
            input_q.put(frame)
        time.sleep(.2)

        if 0xFF == ord('q'):
            break

    input_pool.terminate()
    output_pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
