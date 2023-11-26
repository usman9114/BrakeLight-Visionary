import cv2
import numpy as np
from skimage import measure
from shapely.geometry import Point, Polygon
from helper.constants import (
    NON_ZERO_PIXEL_MAX,
    WHITE_PIXEL_INTENSITY,
    AMBIENT_LIGHT_THRESHOLD,
    LIGHT_INTENSITY_THRESHOLD,
    VEHICLES,
)


class BrakeLightDetect:
    def __init__(self):
        self.black_lower = (0, 0, 0)
        self.black_upper = (180, 255, 35)
        self.hsv_dict = {
            "startRedLowerDay": (0, 160, 150),
            "startRedUpperDay": (10, 255, 255),
            "endRedLowerDay": (170, 180, 90),
            "endRedUpperDay": (180, 255, 255),
            "startRedLowerNight": (0, 165, 150),
            "startRedUpperNight": (10, 255, 255),
            "endRedLowerNight": (170, 175, 90),
            "endRedUpperNight": (180, 255, 255),
        }
        self.white_lower0 = (0, 0, 255)
        self.white_upper0 = (179, 40, 255)
        self.white_lower1 = (0, 100, 255)
        self.white_upper1 = (10, 255, 255)
        self.white_lower2 = (170, 100, 255)
        self.white_upper2 = (180, 255, 255)
        self.num_pixel_day_night = {True: 40, False: 25}
        self.dilate_day_night = {True: 3, False: 6}

        self.brake_flag = False
        self.is_night = False
        self.ambient_light = 0
        self.relative_darkness_score = 0

        self.vehicle_distance = 0

    def getVehicleBbox(self, data_dict):
        x1 = data_dict["x_start"]
        y1 = data_dict["y_start"]
        x2 = x1 + data_dict["width"]
        y2 = y1 + data_dict["height"]

        x1 = int(x1 + data_dict["width"])
        y1 = int(y1 + data_dict["height"])
        x2 = int(x2 - data_dict["width"])
        y2 = int(y2 - data_dict["height"])
        h = data_dict["height"]
        vertices = [
            [x1, y1 - 0.2 * h],
            [x1, y2],
            [x2, y2],
            [x2, y1 - 0.2 * h],
            [x1, y1 - 0.2 * h],
        ]
        return vertices

    def getRoI(self, vertices, img):
        """
        :param vertices: vertices of polygon
        :param img: image from which roi needs to be cropped
        :return: roi
        """
        mask = np.zeros_like(img)
        polygon = np.array([vertices], dtype=np.int32)
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, polygon, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def getCroppedRedArea(self, cnts, img):
        """
        :param cnts: contours from a mask
        :param img: image to apply mask
        :return: returns redlight exposure roi
        """
        boxes = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        boxes = np.asarray(boxes)
        x1, y1 = np.min(boxes, axis=0)[:2]
        x2, y2 = np.max(boxes, axis=0)[2:]

        vertices = [[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]

        return self.getRoI(vertices, img)

    def getContours(self, masked_image, is_night=False):
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)

        if is_night:
            startRedLower = self.hsv_dict["startRedLowerNight"]
            startRedUpper = self.hsv_dict["startRedUpperNight"]
            endRedLower = self.hsv_dict["endRedLowerNight"]
            endRedUpper = self.hsv_dict["endRedUpperNight"]
        else:
            startRedLower = self.hsv_dict["startRedLowerDay"]
            startRedUpper = self.hsv_dict["startRedUpperDay"]
            endRedLower = self.hsv_dict["endRedLowerDay"]
            endRedUpper = self.hsv_dict["endRedUpperDay"]

        mask1 = cv2.inRange(hsv, startRedLower, startRedUpper)
        mask2 = cv2.inRange(hsv, endRedLower, endRedUpper)

        if is_night:
            if self.vehicle_distance <= 5:
                maskRed = mask2
                maskRed = cv2.erode(maskRed, None, iterations=2)
            elif 5 < self.vehicle_distance <= 10:
                maskRed = mask1 + mask2
                maskRed = cv2.erode(maskRed, None, iterations=1)
            else:
                maskRed = mask1 + mask2

            maskRed = cv2.dilate(maskRed, None, iterations=2)
            cnts, hrchy = cv2.findContours(
                maskRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if cnts:
                cropped_red_frame = self.getCroppedRedArea(cnts, hsv)
                # maskRed = cv2.inRange(
                #     cropped_red_frame, self.white_lower, self.white_upper)
                maskRed0 = cv2.inRange(
                    cropped_red_frame, self.white_lower0, self.white_upper0
                )
                maskRed0 = cv2.erode(maskRed0, None, iterations=1)
                maskRed1 = cv2.inRange(
                    cropped_red_frame, self.white_lower1, self.white_upper1
                )
                maskRed2 = cv2.inRange(
                    cropped_red_frame, self.white_lower2, self.white_upper2
                )
                maskRed = maskRed0 + maskRed1 + maskRed2
                maskRed = cv2.dilate(maskRed, None, iterations=1)
        else:
            maskRed = mask1 + mask2
            maskRed = cv2.dilate(
                maskRed, None, iterations=self.dilate_day_night[is_night]
            )
            contours, _ = cv2.findContours(
                maskRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            return contours

        labels = measure.label(maskRed, connectivity=2, background=0)
        mask = np.zeros(maskRed.shape, dtype="uint8")

        if self.vehicle_distance <= 5:
            self.num_pixel_day_night = {True: 40, False: 25}
        elif 5 < self.vehicle_distance <= 10:
            self.num_pixel_day_night = {True: 33, False: 25}
        elif 10 < self.vehicle_distance <= 15:
            self.num_pixel_day_night = {True: 25, False: 25}
        elif 15 < self.vehicle_distance <= 25:
            self.num_pixel_day_night = {True: 10, False: 25}
        else:
            self.num_pixel_day_night = {True: 5, False: 25}

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the # of pixels
            labelMask = np.zeros(maskRed.shape, dtype="uint8")
            labelMask[labels == label] = WHITE_PIXEL_INTENSITY
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if self.num_pixel_day_night[is_night] < numPixels < NON_ZERO_PIXEL_MAX:
                mask = cv2.add(mask, labelMask)

        maskRed = cv2.erode(mask, None, iterations=1)

        if self.vehicle_distance <= 5:
            maskRed = cv2.dilate(
                maskRed, None, iterations=self.dilate_day_night[is_night]
            )
        else:
            maskRed = cv2.dilate(maskRed, None, iterations=1)

        contours, _ = cv2.findContours(
            maskRed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def resetAmbientLightFlags(self):
        """reset the falgs to the default values after each drive completion"""
        self.is_night = False
        self.ambient_light = 0

    def getAmbientLight(self, frame, ambient_light_count):
        """
        :param frame: whole RGB image frame
        :param ambient_light_count: current ambient_light_count
        (initial it is set to 0)
        :return: ambient_light_count
        """
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        pixel_ct = 0
        pixel_len = 0
        for i in mask:
            pixel_ct = pixel_ct + np.sum(i == 0)
            pixel_len = pixel_len + len(i)

        ratio = pixel_ct / pixel_len
        if ratio < LIGHT_INTENSITY_THRESHOLD:
            ambient_light_count = ambient_light_count + 3
        else:
            ambient_light_count = ambient_light_count - 0.5

        return ambient_light_count, ratio

    def getPolygonWidth(self, poly):
        box = poly.minimum_rotated_rectangle
        # get coordinates of polygon vertices
        x, y = box.exterior.coords.xy
        # get length of bounding box edges
        edge_length = (
            Point(x[0], y[0]).distance(Point(x[1], y[1])),
            Point(x[1], y[1]).distance(Point(x[2], y[2])),
        )
        # get width of polygon as the shortest edge of the bounding box
        width = min(edge_length)
        return width

    def getPairAspectRatio(self, p1, p2):
        """
        takes in pair of possible brake light and check symmetry
        aspect ratio check 3< aspect ratio <8
        :param p1: contour1
        :param p2: contour2
        :return: aspect ratio of pair
        """
        p1_width = self.getPolygonWidth(p1)
        p2_width = self.getPolygonWidth(p2)

        return (p1.distance(p2)) / (np.mean([p1_width, p2_width]))

    def breakLightVerificationCheck(self, contours):
        self.brake_flag = False
        possible_brake_lights_pairs = []

        if self.vehicle_distance <= 10:
            pair_ratio_upper = 10
            pair_ratio_lower = 3
        elif 10 < self.vehicle_distance <= 15:
            pair_ratio_upper = 8
            pair_ratio_lower = 2
        else:
            pair_ratio_upper = 5
            pair_ratio_lower = 0.1

        for ci in range(len(contours)):
            try:
                poly_i = Polygon(np.squeeze(contours[ci]))
                if not poly_i.is_valid:
                    poly_i = poly_i.buffer(0)

                for cj in range(ci + 1, len(contours)):
                    poly_j = Polygon(np.squeeze(contours[cj]))
                    if not poly_j.is_valid:
                        poly_j = poly_j.buffer(0)
                    pair_aspect_ratio = self.getPairAspectRatio(poly_i, poly_j)
                    if poly_i.symmetric_difference(poly_j) and (
                        pair_ratio_lower <= pair_aspect_ratio <= pair_ratio_upper
                    ):
                        possible_brake_lights_pairs.append(ci)
                        possible_brake_lights_pairs.append(cj)

            except Exception:
                pass

        return possible_brake_lights_pairs

    def detect(self, contours):
        possible_brake_lights_pairs = self.breakLightVerificationCheck(contours)

        if possible_brake_lights_pairs:
            self.brake_flag = True

            boxes = []
            for c in possible_brake_lights_pairs:
                (x, y, w, h) = cv2.boundingRect(contours[c])
                boxes.append([x, y, x + w, y + h])

            boxes = np.asarray(boxes)
            x1, y1 = np.min(boxes, axis=0)[:2]
            x2, y2 = np.max(boxes, axis=0)[2:]

            return (
                True,
                {
                    "x_start": int(x1),
                    "y_start": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
            )

        else:
            return False, {}

    def get_ambient_light(self):
        """This method returns the ambient light score and a flag to check day/night

        Returns:
            dict: ambient light score -> {
                "relative_darkness_score": 0.0,
                "is_light": True/False,
            }
        """
        return {
            "relative_darkness_score": self.relative_darkness_score,
            "is_light": not self.is_night,
        }

    def detect_brake_light(self, objects, img_path):
        """This method detetcts brake light (if brake_light deteted then add bounding
               box into the object dictonary

        Args:
            objects (list): list of objects detected in an image, objects -> [
                {
                    class (str): <class-name> of detected object
                    confidence (int): confidance of predection
                    x_start (int): bounding box x-start
                    y_start (int): bounding box y-start
                    width (int):  bounding box width
                    height (int): bounding box height
                }, ...
            ]
            img_path (str): complete path to an image

        Returns:
            list: lit of objects with brake_lights, resp_obj -> [
                {
                    class (str): <class-name> of detected object
                    confidence (int): confidance of predection
                    x_start (int): bounding box x-start
                    y_start (int): bounding box y-start
                    width (int):  bounding box width
                    height (int): bounding box height
                    brake_light (dict): add brake_light key if brake light detected, brake_light -> {
                        "bounding_box": {
                            "x_start": 0,
                            "y_start": 0,
                            "width": 0,
                            "height": 0
                        }
                    }
                }, ...
            ]
        """

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.ambient_light, self.relative_darkness_score = self.getAmbientLight(
            img, self.ambient_light
        )

        if self.ambient_light > AMBIENT_LIGHT_THRESHOLD:
            self.is_night = True

        for obj in objects:
            if obj["class"] in VEHICLES:

                self.vehicle_distance = obj.get("calculated_distance_in_meters", 0)

                vertices = self.getVehicleBbox(obj)
                masked = self.getRoI(vertices, img)
                contours = self.getContours(masked, self.is_night)
                brake_detected, b_box = self.detect(contours)

                if brake_detected:
                    obj["brake_lights"] = {"bounding_box": b_box}

        return objects
