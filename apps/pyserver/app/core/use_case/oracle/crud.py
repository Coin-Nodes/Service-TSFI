from six import BytesIO
import requests
import numpy as np
import cv2

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import tempfile
import time
import os
import io



class CRUDOracle:

    @staticmethod
    def get_image_as_array(image):
        image_data = BytesIO(image)
        file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        return img_array

    @staticmethod
    def make_request(image, server_url):
        """ Send request to the Tensorflow container """
        img_array = CRUDOracle.get_image_as_array(image)
        # reshaping using cv2 instead of numpy as suggested in:
        # https://github.com/tensorflow/models/issues/2503
        np_image = np.expand_dims(img_array, 0).tolist()
        request_data = '{"instances" : %s}' % np_image

        r = requests.post(server_url, data=request_data)

        return r.json()

    @staticmethod
    def get_predictions(image, server_url):
        """ Get the filtered Predictions key from the TensorFlow request """
        predictions = CRUDOracle.make_request(image, server_url)["predictions"][0]

        classes_names = CRUDOracle.get_classnames_dict()
        num_detections = int(predictions["num_detections"])
        # Filtering out the unused predictions
        detection_boxes = predictions["detection_boxes"][:num_detections]
        detection_classes = predictions["detection_classes"][:num_detections]
        detection_classes_names = []
        for detection in detection_classes:
            detection_classes_names.append(classes_names[detection - 1])
        detection_scores = predictions["detection_scores"][:num_detections]

        return {"num_detections": num_detections,
                "detection_boxes": detection_boxes,
                "detection_classes": detection_classes_names,
                "detection_scores": detection_scores}

    @staticmethod
    def draw_bounding_box_on_image(image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color,
                                   font,
                                   thickness=4,
                                   display_str_list=()):
        """Adds a bounding box to an image."""
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=thickness,
                  fill=color)

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                            (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill="black",
                      font=font)
            text_bottom -= text_height - 2 * margin

    @staticmethod
    def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
        """Overlay labeled boxes on an image with formatted scores and label names."""
        colors = list(ImageColor.colormap.values())

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
        except IOError:
            print("Font not found, using default font.")
            font = ImageFont.load_default()

        for i in range(min(len(boxes), max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                # display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                #                                 int(100 * scores[i]))
                display_str = "{}: {}%".format(class_names[i], int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                CRUDOracle.draw_bounding_box_on_image(image_pil,
                                           ymin,
                                           xmin,
                                           ymax,
                                           xmax,
                                           color,
                                           font,
                                           display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
        return image

    @staticmethod
    def save_img(image):
        """ Save the image file locally"""
        _, filename = tempfile.mkstemp(suffix=".jpg")
        im = Image.fromarray(image)
        im.save(filename)

        return filename

    @staticmethod
    def get_classnames_dict():
        """ Get the classes instances from the COCO dataset """
        classes = {}
        i = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        classes_file = open(dir_path + "/coco-labels-paper.txt")
        for line in classes_file:
            classes[i] = line.split("\n")[0]
            i += 1

        return classes

    @staticmethod
    def get_predicted_image(image, server_url, detections_limit):
        """ Run the prediction, draw boxes around the objects in the image and
        return the image as file """
        start_time = time.time()
        img = CRUDOracle.get_image_as_array(image)
        end_time = time.time()

        result = CRUDOracle.get_predictions(image, server_url)

        print("Found %d objects." % result["num_detections"])
        print("Inference time: ", end_time - start_time)

        image_with_boxes = CRUDOracle.draw_boxes(img, result["detection_boxes"],
                                      result["detection_classes"],
                                      result["detection_scores"],
                                      detections_limit)

        filename = CRUDOracle.save_img(image_with_boxes)

        f = io.BytesIO()
        f = open(filename, 'rb')
        content = f.read()
        f.close()

        return content