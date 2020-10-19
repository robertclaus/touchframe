import cv2
import matplotlib.pyplot as plt
from SignalDetector import SignalDetector
from Mapper import  Mapper


class Camera:
    def __init__(self, hardware_id, position_x, position_y, angle, viewing_angle):
        self.hardware_id = hardware_id
        self.camera_reader = SignalDetector(self.hardware_id)
        self.position_x = position_x
        self.position_y = position_y
        self.angle = angle
        self.viewing_angle = viewing_angle


cameras = [
    Camera(2, 250, 10, 0, 60),
    Camera(3, 490, 10, -45, 60),
    #Camera(4, 10, 10, 45, 60),
]

mapper = Mapper()


while(True):
    mapper.reset_plot()
    process_results = []

    for camera in cameras:
        process_result = camera.camera_reader.process()
        if process_result:
            mapper.add_signal_with_camera(process_result, camera)
        process_results.append(process_result)

    mapper.draw_image()

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('c'):
        [camera.camera_reader.calibrate() for camera in cameras]
    if k == ord('p'):
        fig, axs = plt.subplots(len(process_results))
        for idx, output_signal in enumerate(process_results):
            if len(process_results) > 1:
                axs[idx].scatter(range(len(output_signal)), output_signal)
            else:
                axs.scatter(range(len(output_signal)), output_signal)
        fig.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
releases = [camera.camera_reader.release() for camera in cameras]
cv2.destroyAllWindows()