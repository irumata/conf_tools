import argparse
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils import video

CROP_SIZE = 256
DOWNSAMPLE_RATIO = 2


class AvatarProcessor():

    def __init__(self, frozen_graph_filename):
        self.graph = self.load_graph(frozen_graph_filename)
        self.sess = tf.Session(graph=self.graph)
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.output_tensor = self.graph.get_tensor_by_name('generate_output/output:0')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor
        #self.predictor = dlib.shape_predictor(args.face_landmark_shape_file)
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def process_frame(self, frame):
        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        black_image = np.zeros(frame.shape, np.uint8)

        for face in faces:
            detected_landmarks = self.predictor(gray, face).parts()
            landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

            jaw = self.reshape_for_polyline(landmarks[0:17])
            left_eyebrow = self.reshape_for_polyline(landmarks[22:27])
            right_eyebrow = self.reshape_for_polyline(landmarks[17:22])
            nose_bridge = self.reshape_for_polyline(landmarks[27:31])
            lower_nose = self.reshape_for_polyline(landmarks[30:35])
            left_eye = self.reshape_for_polyline(landmarks[42:48])
            right_eye = self.reshape_for_polyline(landmarks[36:42])
            outer_lip = self.reshape_for_polyline(landmarks[48:60])
            inner_lip = self.reshape_for_polyline(landmarks[60:68])

            color = (255, 255, 255)
            thickness = 3

            cv2.polylines(black_image, [jaw], False, color, thickness)
            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            cv2.polylines(black_image, [lower_nose], True, color, thickness)
            cv2.polylines(black_image, [left_eye], True, color, thickness)
            cv2.polylines(black_image, [right_eye], True, color, thickness)
            cv2.polylines(black_image, [outer_lip], True, color, thickness)
            cv2.polylines(black_image, [inner_lip], True, color, thickness)

        combined_image = np.concatenate([self.resize(black_image), self.resize(frame_resize)], axis=1)
        image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR instead of RGB
        generated_image = self.sess.run(self.output_tensor, feed_dict={self.image_tensor: image_rgb})
        image_bgr = cv2.cvtColor(np.squeeze(generated_image), cv2.COLOR_RGB2BGR)

        return image_bgr


    def reshape_for_polyline(self, array):
        """Reshape image so that it works with polyline."""
        return np.array(array, np.int32).reshape((-1, 1, 2))

    def resize(self, image):
        """Crop and resize image for pix2pix."""
        height, width, _ = image.shape
        if height != width:
            # crop to correct ratio
            size = min(height, width)
            oh = (height - size) // 2
            ow = (width - size) // 2
            cropped_image = image[oh:(oh + size), ow:(ow + size)]
            image_resize = cv2.resize(cropped_image, (CROP_SIZE, CROP_SIZE))
            return image_resize

    def load_graph(self, frozen_graph_filename):
        """Load a (frozen) Tensorflow model into memory."""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('--show', dest='display_landmark', type=int, default=0, choices=[0, 1],
                        help='0 shows the normal input and 1 the facial landmark.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str,
                        help='Face landmark model file.')
    parser.add_argument('--tf-model', dest='frozen_model_file', type=str, help='Frozen TensorFlow model file.')

    args = parser.parse_args()

    avatar = AvatarProcessor(args.frozen_model_file)

    cap = cv2.VideoCapture(args.video_source)
    fps = video.FPS().start()
    while True:
        ret, frame = cap.read()
        ret_img = avatar.process_frame(frame)
        cv2.imshow('frame', ret_img)
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()
