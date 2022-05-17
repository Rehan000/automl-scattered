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


def show_values_forever():
    try:
        tracker = MultiObjectTracker(dt=0.1)
        tracking_ids = {}
        while True:
            message = redis.xread({stream_name: '$'}, None, 0)
            # image_shape_string = message[0][1][0][1][b"shape"]
            encoded_image = message[0][1][0][1][b'tracking']
            bbox_string = message[0][1][0][1][b"bbox"]
            bbox_size = np.fromstring(message[0][1][0][1][b"bbox_shape"], dtype=int)
            bbox = np.fromstring(bbox_string, dtype=np.float64)
            detections_bs = bbox.reshape(bbox_size)
            # print(detections_bs)
            nparr = np.fromstring(encoded_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            detections = [Detector({"box": x[1:5], 'score': x[5], 'class_id': x[6], 'feature': 0})
                          for x in detections_bs.tolist()]
            if len(detections) != 0:
                tracker.step(detections=detections)
                tracked = tracker.active_tracks()
                # for track in tracked:
                #     if len(list(tracking_ids.keys()))==0:
                #         tracking_ids[track["id"]] = 0
                #     else:
                #         if track["id"] not in tracking_ids:
                #             tracking_ids[track["id"]] = max(tracking_ids.values())+1

                detections = [[detections_bs[i, 0]] + tracked[i].box.tolist() + [tracked[i].score] + [tracked[i].id]
                              for i in range(len(detections))]
                prediction = np.array(detections)

                tracked_img = visualize_image_prediction(img, prediction, True)
                cv2.imshow("tracked_img", tracked_img)
                cv2.waitKey(1)
            # img = visualize_image_prediction(img, reshaped_detection, False)
            # cv2.imshow("received_image", img)
            # cv2.waitKey(1)
            # print('*** show_values_forever :: message -> {}'.format(message))
    except Exception as ex:
        print(ex)
    print('show_values terminated.')


show_values_forever()