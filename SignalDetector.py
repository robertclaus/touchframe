import math
import cv2
import numpy as np

offset = 10


class SignalDetector:
    def __init__(self, camera_num, calibration_width=1000, calibration_height=500, measurement_offset=10, thresh_min=30):
        """ Sets up the camera object.

        Calibration refers to the frame the camera captured.  This is used for drawing the calibration line.
        Measurement refers to the data along the calibration line
        Output refers to the actual signal we want to process

        camera_num : The integer index to the correct camera on your machine.
        window_name : Name of the window for this camera
        calibration_width : Width of the main calibration image display
        calibration_height : Height of the main calibration image display
        measurement_offset : Height of the measurment image below the calibration image

        """
        self.camera_num = camera_num
        window_name = "Camera " + str(camera_num)
        self.cap = cv2.VideoCapture(camera_num)
        cv2.namedWindow(window_name)
        self.calibration_line = [None, None]
        self.calibrating = False
        cv2.setMouseCallback(window_name, self.create_click_callback())

        self.calibration_width = calibration_width
        self.calibration_height = calibration_height
        self.measurement_offset = measurement_offset
        self.threshold_minimum = thresh_min

        self.window_name = window_name
        self.previous_measurement = np.zeros((self.measurement_offset, self.calibration_width, 3), np.uint8)
        self.current_measurement = np.zeros((self.measurement_offset, self.calibration_width, 3), np.uint8)
        self.output = np.zeros((self.measurement_offset, self.calibration_width, 3), np.uint8)

    def process(self):
        """ Gets and processes a frame. """
        ret, raw_frame = self.cap.read()
        frame = cv2.resize(raw_frame, (self.calibration_width, self.calibration_height))

        # Default measurement frame to zeroes
        new_measurement = np.zeros((self.measurement_offset, self.calibration_width, 3), np.uint8)

        # Calculate new measurement frame based on the calibration line
        if not self.calibrating and self.calibration_line[0] and self.calibration_line[1]:
            height = float(self.calibration_line[1][1] - self.calibration_line[0][1])
            width = float(self.calibration_line[1][0] - self.calibration_line[0][0])

            angle = math.atan(height / width) * 180 / 3.14
            length = math.sqrt(math.pow(height, 2) + math.pow(width, 2))

            rotated = SignalDetector.rotate_image(frame, angle, center=self.calibration_line[0])
            cropped = rotated[
                      int(self.calibration_line[0][1] - offset): int(self.calibration_line[0][1] + offset),
                      int(self.calibration_line[0][0] - offset): int(self.calibration_line[0][0] + length + offset),
                      ]
            new_measurement = cv2.resize(cropped, (self.calibration_width, self.measurement_offset))

        # Draw calibration lines on the new frame
        if self.calibration_line[0] and self.calibration_line[1]:
            cv2.line(frame, self.calibration_line[0], self.calibration_line[1], (0, 255, 0), 2)
        if self.calibration_line[0]:
            cv2.circle(frame, self.calibration_line[0], 5, (255, 0, 0))
        if self.calibration_line[1]:
            cv2.circle(frame, self.calibration_line[1], 5, (255, 0, 0))

        # Detect objects based on the new frame
        self.detect(new_measurement)

        # Updates the state of the detector based on the new measurement
        self.update_state(new_measurement)

        cv2.imshow(self.window_name, np.vstack((frame, new_measurement, self.previous_measurement, self.output)))
        cv2.moveWindow(self.window_name, (self.camera_num % 1) * self.calibration_width, (((self.camera_num / 2) % 2) * self.calibration_height) )

        return self.get_values()

    def detect(self, new_measurement):
        """ Process the new measurement into an output. """
        output = cv2.absdiff(self.previous_measurement, new_measurement)
        self.output = output

    def update_state(self, new_measurement):
        """ Updates any internal state for the camera such as the latest background. """
        # self.previous_measurement = cv2.addWeighted(new_measurement, 0.01, self.previous_measurement, 0.99, 1)
        self.current_measurement = new_measurement

    def release(self):
        """ Closes the connection to the camera."""
        self.cap.release()

    @staticmethod
    def rotate_image(image, angle, center=None, scale=1.0):
        """ Rotates an image. """
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def create_click_callback(self):
        """ Returns a click callback function that is bound to self. """
        def click_callback(event, x, y, flags, param):
            """ Handles a click on the current frame. """

            if event == cv2.EVENT_LBUTTONDOWN:
                self.calibration_line[0] = (x, y)
                self.calibrating = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.calibration_line[1] = (x, y)
                self.calibrating = False
                if self.calibration_line[0][0] == self.calibration_line[1][0]:
                    self.calibration_line[1] = None
                else:
                    self.process()
                    self.calibrate()
            elif event == cv2.EVENT_MOUSEMOVE and self.calibrating:
                self.calibration_line[1] = (x, y)
        return click_callback

    def get_values(self):
        gray_output = cv2.cvtColor(self.output, cv2.COLOR_BGR2GRAY)
        rows, cols = gray_output.shape
        threshold_output = cv2.threshold(gray_output, 30.0, 255.0, type=cv2.THRESH_TOZERO)[1]
        cv2.normalize(threshold_output, threshold_output, 0, 80, cv2.NORM_MINMAX)
        output_signal = [max(threshold_output[:, c]) for c in range(cols)]
        return output_signal

    def calibrate(self):
        self.previous_measurement = self.current_measurement
