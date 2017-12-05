import numpy as np
import os
import sys
import tensorflow as tf
import argparse
import cv2
from collections import defaultdict
from multiprocessing import Queue, Pool
import time
from imutils.video import FPS
import logging
sys.path.append('/models/research/object_detection/')
from utils import label_map_util
from utils import visualization_utils as vis_util
from slackclient import SlackClient

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/model_zoo/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/models/research/object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def detect_objects(image_np, sess, detection_graph, min_score_thresh):
    logging.debug('running detection')
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    scores_squeezed = np.squeeze(scores)
    classes_squeezed = np.squeeze(classes).astype(np.int32)
    #logging.debug(scores_squeezed)
    #logging.debug(classes_squeezed)
    for i in range(len(scores_squeezed)):
        if scores_squeezed[i] >= min_score_thresh and classes_squeezed[i] == 1:
            break
    else:
        return False

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        classes_squeezed,
        scores_squeezed,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=min_score_thresh,
        line_thickness=8)
    print('detected')
    return True

def detector_worker(input_q, output_q, min_score_thresh):
    print('tensorflow init')
    # init tensorflow
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
    print('tensorflow init done')
    while True:
        logging.debug('getting pic')
        frame = input_q.get()
        logging.debug('got pic')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        object_detected = detect_objects(frame_rgb, sess, detection_graph, min_score_thresh)
        if object_detected:
            output_q.put(frame_rgb)

def notifier_worker(output_q, slack_token):
    # init Slack
    sc = SlackClient(args.slack_token)
    while True:
        frame_rgb = output_q.get()
        logging.debug('slack')
        cv2.imwrite("image.jpg", frame_rgb)
        in_file = open("image.jpg", "rb") # opening for [r]eading as [b]inary
        #data = in_file.read()
        res = sc.api_call("files.upload", filename="image.jpg", channels="#eagle_eye", file=in_file)
        logging.debug(res)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        help='rtsp://admin:*@192.*.*.*:554/h264/ch1/main/av_stream')
    parser.add_argument('-token', '--slack_token', dest='slack_token', type=str,
                        help='Slack Web API Token')
    parser.add_argument('-channel', '--slack_channel', dest='slack_channel', type=str,
                        help='Slack Web API Token')
    parser.add_argument('-min_score_thresh', '--min_score_thresh', dest='min_score_thresh', type=float,
                        default=0.5, help='percentage as min confidence score')
    parser.add_argument('-sleep_time', '--sleep_time', dest='sleep_time', type=float,
                        default=0.3, help='sleep time between each frame') 
    args = parser.parse_args()

    input_q = Queue(maxsize=50)
    output_q = Queue(maxsize=50)
    detector_pool = Pool(2, detector_worker, (input_q, output_q, args.min_score_thresh))
    notifier_pool = Pool(1, notifier_worker, (output_q, args.slack_token))

    # init OpenCV
    video_capture = cv2.VideoCapture(args.video_source)

    fps = FPS().start()

    while(1):
        ret, frame = video_capture.read()
        if ret:
            logging.debug(input_q.qsize())
            input_q.put(frame)
            logging.debug('putting pic')
            time.sleep(args.sleep_time)
        else:
            print('video open')
            video_capture.open(args.video_source)
        if 0xFF == ord('q'):
            break
        fps.update()
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
    sess.close()
    detector_pool.terminate()
    notifier_pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
