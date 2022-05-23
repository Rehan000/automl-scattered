from inference import visualize_image_prediction
from motpy import MultiObjectTracker
import redis

import numpy as np
import cv2

redis = redis.Redis(host='127.0.0.1')

stream_name = 'detection'


class Detector:
    def __init__(self, object_dict):
        self.box = object_dict["box"]
        self.score = object_dict['score']
        self.class_id = object_dict['class_id']
        self.feature = object_dict['feature']


def crop_bbox(num_bbox, bbox_list, fullsized_decoded_image):
    print("Number of bounding boxes:", num_bbox)
    print("Bounding boxes:", bbox_list)

    '''
    rx = 3840 / 500
    ry = 2160 / 281
    total_bbox_list = []
    for i in range(0, num_bbox * 7, 7):
        y_min, x_min, y_max, x_max = np.array(bbox_list[i+1:i+5], dtype=int)
        y_min = y_min * ry
        x_min = x_min * rx
        y_max = y_max * ry
        x_max = x_max * rx
        total_bbox_list.append([y_min, x_min, y_max, x_max])

    total_bbox_list = np.array(total_bbox_list, dtype=int)
    print()
    '''


def show_values_forever():
    try:
        # Create tracker object
        tracker = MultiObjectTracker(dt=0.1)
        # List to store tracking ids
        tracking_ids = []
        while True:
            # Read the redis stream: "detection"
            message = redis.xread({stream_name: '$'}, None, 0)

            # Extract the detection parameters
            fullsized_encoded_image = message[0][1][0][1][b'fullsized_image']
            resized_encoded_image = message[0][1][0][1][b'resized_image']
            bbox_string = message[0][1][0][1][b"bbox"]
            num_bbox = np.frombuffer(message[0][1][0][1][b"num_bbox"], dtype=int)
            bbox = np.frombuffer(bbox_string, dtype=np.float64)
            detections_bs = bbox.reshape(num_bbox)

            # Convert image from base_64 to normal format
            resized_decoded_image = np.frombuffer(resized_encoded_image, np.uint8)
            resized_decoded_image = cv2.imdecode(resized_decoded_image, cv2.IMREAD_COLOR)
            fullsized_decoded_image = np.frombuffer(fullsized_encoded_image, np.uint8)
            fullsized_decoded_image = cv2.imdecode(fullsized_decoded_image, cv2.IMREAD_COLOR)

            crop_bbox(num_bbox=num_bbox[0], bbox_list=bbox, fullsized_decoded_image=fullsized_decoded_image)

            # Reformat detections
            detections = [Detector({"box": x[1:5], 'score': x[5], 'class_id': x[6], 'feature': 0})
                          for x in detections_bs.tolist()]

            # If detections != 0 then get their tracking ids
            if len(detections) != 0:
                tracker.step(detections=detections)
                tracked = tracker.active_tracks()
                detections = [[detections_bs[i, 0]] + tracked[i].box.tolist() + [tracked[i].score] + [tracked[i].id]
                              for i in range(len(detections))]
                prediction = np.array(detections)

                '''
                for pred in prediction[:, -1]:
                    if pred not in tracking_ids:
                        tracking_ids.append(pred)
                        print("New tracking id added:", pred)
                        print("Total tracking ids:", len(tracking_ids))
                    else:
                        print("No new tracking id.")
                        print("Total tracking ids:", len(tracking_ids))
                '''

                # Visualize the image with tracking ids
                tracked_img = visualize_image_prediction(resized_decoded_image, prediction, True)
                cv2.imshow("Tracking Image", tracked_img)
                cv2.waitKey(1)
    except Exception as ex:
        print(ex)
    print('show_values terminated.')


show_values_forever()