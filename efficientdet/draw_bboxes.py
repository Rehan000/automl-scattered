from inference import visualize_image_prediction
import redis

import numpy as np
import cv2

redis = redis.Redis(host='127.0.0.1')

stream_name = 'detection'


def show_values_forever():
    try:
        while True:
            message = redis.xread({stream_name: '$'}, None, 0)
            # image_shape_string = message[0][1][0][1][b"shape"]
            print(message[0][1][0][1].keys())
            encoded_image = message[0][1][0][1][b'tracking']
            bbox_string = message[0][1][0][1][b"bbox"]
            bbox_size = np.fromstring(message[0][1][0][1][b"bbox_shape"], dtype=int)
            bbox = np.fromstring(bbox_string, dtype=np.float64)
            reshaped_detection = bbox.reshape(bbox_size)
            nparr = np.fromstring(encoded_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(reshaped_detection)
            img = visualize_image_prediction(img, reshaped_detection, False)
            cv2.imshow("received_image", img)
            cv2.waitKey(1)
            # print('*** show_values_forever :: message -> {}'.format(message))
    except Exception as ex:
        print(ex)
    print('show_values terminated.')


show_values_forever()