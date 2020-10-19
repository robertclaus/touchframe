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
releases = [camera.release() for camera in cameras]
cv2.destroyAllWindows()