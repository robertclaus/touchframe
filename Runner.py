import cv2
import matplotlib.pyplot as plt
from SignalDetector import SignalDetector

camera_ids = [1]
cameras = [SignalDetector(camera_id) for camera_id in camera_ids]

while(True):
    process_results = [camera.process() for camera in cameras]

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('c'):
        [camera.calibrate() for camera in cameras]
    if k == ord('p'):
        for output_signal in process_results:
            plt.scatter(range(len(output_signal)), output_signal)
            plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
releases = [camera.release() for camera in cameras]
cv2.destroyAllWindows()