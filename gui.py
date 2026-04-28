import cv2
import tkinter as tk
from PIL import Image, ImageTk

from utils import get_hand_contour
from model import SignLanguageModel

# Load model
model = SignLanguageModel()

# GUI window
root = tk.Tk()
root.title("Sign Language Detection")
root.geometry("800x600")

# Label to show video
label = tk.Label(root)
label.pack()

# Webcam
cap = cv2.VideoCapture(0)

def update():

    ret, frame = cap.read()
    if not ret:
        root.after(10, update)
        return

    # Flip frame (mirror view)
    frame = cv2.flip(frame, 1)

    # Get hand contour
    contour = get_hand_contour(frame)

    # Prediction
    prediction = model.predict(contour)

    # Draw contour if exists
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # Show prediction text
    cv2.putText(frame,
                prediction,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3)

    # Debug window (optional but helpful)
    cv2.imshow("Live Camera", frame)

    # Convert for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Repeat loop
    root.after(10, update)

def on_close():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

update()
root.mainloop()