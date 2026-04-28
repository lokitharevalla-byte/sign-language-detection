import cv2
import numpy as np

class SignLanguageModel:

    def predict(self, contour):

        if contour is None:
            return "NO HAND DETECTED"

        hull = cv2.convexHull(contour, returnPoints=False)

        if hull is None or len(hull) < 3:
            return "UNKNOWN"

        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            return "FIST ✊"

        count = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))

            if b * c == 0:
                continue

            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

            if angle < 1.2:  # finger gap
                count += 1

        if count == 0:
            return "FIST ✊"
        elif count == 1:
            return "ONE ☝️"
        elif count == 2:
            return "TWO ✌️"
        else:
            return "OPEN HAND 🖐"