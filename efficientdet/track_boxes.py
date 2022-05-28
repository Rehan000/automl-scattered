from inference import visualize_image_prediction
from motpy import MultiObjectTracker
import redis
import base64
import codecs
import rekognition
import numpy as np
import cv2

redis = redis.Redis(host='127.0.0.1')
rekognition_client = rekognition.Rekognition()

stream_name = 'detection'

tracking_dict = {}


class Detector:
    def __init__(self, object_dict):
        self.box = object_dict["box"]
        self.score = object_dict['score']
        self.class_id = object_dict['class_id']
        self.feature = object_dict['feature']


def crop_bbox(fullsized_decoded_image, tracking_predictions,
              collection_id="FacesCollection"):
    resize_ratio = 7.68
    bbox_crop_list = []
    total_bbox_list = []

    for prediction in tracking_predictions:
        y_min = int(float(prediction[1]))
        x_min = int(float(prediction[2]))
        y_max = int(float(prediction[3]))
        x_max = int(float(prediction[4]))
        y_min = y_min * resize_ratio
        x_min = x_min * resize_ratio
        y_max = y_max * resize_ratio
        x_max = x_max * resize_ratio
        total_bbox_list.append([y_min, x_min, y_max, x_max])

    total_bbox_list = np.array(total_bbox_list, dtype=int)
    for bbox in total_bbox_list:
        cropped_image = fullsized_decoded_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        bbox_crop_list.append(cropped_image)

    for index in range(len(tracking_predictions[:, -1])):
        if tracking_predictions[:, -1][index] not in tracking_dict.values():
            cv2.imwrite("cropped_image.jpg", bbox_crop_list[index])
            try:
                response = rekognition_client.search_face(collection_id=collection_id,
                                                          source_image="cropped_image.jpg")
                print("AWS Request Sent!")
                if len(response['FaceMatches']) != 0:
                    print("Face Matched!")
                    tracking_dict[response['FaceMatches'][0]['Face']['ExternalImageId']] = \
                        tracking_predictions[:, -1][index]
                else:
                    print("No Face Matched!")
            except rekognition_client.client.exceptions.InvalidParameterException:
                pass


def show_values_forever():
    CALL_CROP_BOX = False
    try:
        # Create tracker object
        tracker = MultiObjectTracker(dt=60.0)
        # Set to store tracking ids
        tracking_ids_set = set()
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

            # Reformat detections
            detections = [Detector({"box": x[1:5], 'score': x[5], 'class_id': x[6], 'feature': 0})
                          for x in detections_bs.tolist()]

            # If detections != 0 then get their tracking ids
            if len(detections) != 0:
                tracker.step(detections=detections)
                tracked = tracker.active_tracks()
                detections = [[detections_bs[i, 0]] + tracked[i].box.tolist() + [tracked[i].score] + [tracked[i].id]
                              for i in range(len(detections))]
                tracking_predictions = np.array(detections)

                # If new tracking id, add in tracking id list and call crop_bbox function
                for tracking_id in tracking_predictions[:, -1]:
                    if tracking_id not in tracking_ids_set:
                        tracking_ids_set.add(tracking_id)
                        CALL_CROP_BOX = True

                if CALL_CROP_BOX:
                    crop_bbox(fullsized_decoded_image=fullsized_decoded_image,
                              tracking_predictions=tracking_predictions)
                    CALL_CROP_BOX = False

            # Visualize the image with tracking ids
            if len(detections) != 0:
                tracked_img = visualize_image_prediction(resized_decoded_image, np.array(detections), True)
                tracked_img = cv2.resize(tracked_img, (800, 500))
                cv2.imshow("Tracking Image", tracked_img)
                cv2.waitKey(1)
            else:
                tracked_img = resized_decoded_image
                tracked_img = cv2.resize(tracked_img, (800, 500))
                cv2.imshow("Tracking Image", tracked_img)
                cv2.waitKey(1)

            print("Tracking Dictionary:", tracking_dict)

    except Exception as ex:
        print(ex)
        print('show_values terminated.')


show_values_forever()
