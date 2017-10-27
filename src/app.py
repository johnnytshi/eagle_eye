import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import multiprocessing
from multiprocessing import Queue, Pool
from collections import defaultdict
from io import StringIO

sys.path.append('/models/research/object_detection/')
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/models/research/object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index);

def detect_objects(image_np, sess, detection_graph):
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

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=.1,
        line_thickness=8)
    return image_np

def input_worker(input_q, output_q):
    count = 0;
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    while True:
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #output_q.put(detect_objects(frame_rgb, sess, detection_graph))
        output_rgb = detect_objects(frame_rgb, sess, detection_graph)
        cv2.imwrite("images/image{0}.jpg".format(count), output_rgb)
        count += 1

    sess.close()


if __name__ == '__main__':
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(24)
    output_q = Queue(24)
    pool = Pool(1, input_worker, (input_q, output_q))

    video_capture = cv2.VideoCapture("rtsp://admin:***@192.168.2.15:554/h264/ch2/main/av_stream")
    frame_counter = 0;
    while(1):
        ret, frame = video_capture.read()

        if frame_counter%10 == 0:
            input_q.put(frame)
        frame_counter += 1

        if 0xFF == ord('q'):
            break

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
