import cv2
import numpy as np
import math


class Mapper:
    def __init__(self):
        self.L = 5000
        self.w = 500
        self.h = 500
        self.image = self.reset_plot()

    def add_signal(self, signal, cx, cy, ca, va):
        """

        :param signal: Array of values to plot
        :param cx: X position of the camera
        :param cy: Y position of the camera
        :param ca: Angle of the camera (degrees)
        :param va: Viewing angle of the camera (in degrees)
        :return:
        """
        temp_image = self.get_empty_image()

        angle_width = float(va) / len(signal)
        for idx, point in enumerate(signal):
            contour = self.calculate_triangle(cx, cy, ca, angle_width*idx, angle_width)

            cv2.drawContours(
                temp_image,
                np.array([contour]),
                0,
                (point),
                -1
            )
        self.image = cv2.addWeighted(temp_image, 1, self.image, 1, 0)

    def calculate_triangle(self, cx, cy, ca, ma, angle_width):
        """

        :param cx: X position of the camera
        :param cy: Y position of the camera
        :param ca: Angle of the camera (degrees)
        :param ma: Angle measured by the camera
        :param degrees: Degrees this triangle covers
        :return:
        """

        ca = math.radians(ca)
        ma = math.radians(ma)
        angle_width = math.radians(angle_width)

        point_a = (cx, cy)
        point_b = (int(cx + self.L*math.sin(ca + ma - angle_width)), int(cy + self.L*math.cos(ca + ma - angle_width)))
        point_c = (int(cx + self.L*math.sin(ca + ma + angle_width)), int(cy + self.L*math.cos(ca + ma + angle_width)))

        return [point_a, point_b, point_c]

    def get_empty_image(self):
        return np.zeros((self.w, self.h, 1), np.uint8)

    def reset_plot(self):
        self.image = self.get_empty_image()
        return self.image

    def draw_image(self):
        window_name = "Projected Output"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, self.image)


signal_1 = [25]*10 + [50]*10 + [75]*10 + [100]*10
signal_2 = [25]*10 + [50]*10 + [75]*10 + [100]*10

mapper = Mapper()
mapper.add_signal(signal_1, 100, 100, 0, 45)
mapper.add_signal(signal_2, 300, 100, -45, 60)
mapper.draw_image()

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()