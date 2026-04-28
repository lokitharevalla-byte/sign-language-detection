import cv2
import numpy as np

def get_hand_contour(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # adaptive threshold (better than fixed threshold)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        10
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # take largest contour BUT ignore tiny noise
    max_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(max_contour) < 3000:
        return None

    return max_contour