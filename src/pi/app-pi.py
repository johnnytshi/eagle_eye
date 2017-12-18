import tensorflow as tf
import sys
from multiprocessing import Queue, Pool
import cv2
from slackclient import SlackClient

def detect_objects(sess, softmax_tensor, image_data, min_score_thresh):
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

def detector_worker(input_q, output_q, min_score_thresh):
    print('tensorflow init')
    # init tensorflow
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("output_labels.txt")]
    # Unpersists graph from file
    with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    print('tensorflow init done')
    while True:
        logging.debug('getting pic')
        frame = input_q.get()
        logging.debug('got pic')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_objects(sess, softmax_tensor, frame_rgb, min_score_thresh)
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
