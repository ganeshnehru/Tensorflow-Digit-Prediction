import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from keras.models import load_model


# Load the model.
model = load_model('model.h5')


# Select video file.
def openFile():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


# Get user option input. 
def getOption(user_input):
    # Set up internal webcam.
    if user_input == 1:
        return cv2.VideoCapture(0)
    # Set up external webcam.
    elif user_input == 2:
        return cv2.VideoCapture(1)
    # Set up selected video file.
    elif user_input == 3:
        return cv2.VideoCapture(openFile())
    else:
        return IOError("Invalid input.")


# Merges all the images from the frame.
def merge_images(images):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_height = max(heights)
    total_width = sum(widths)
    merged = np.zeros((max_height, total_width), dtype=np.uint8)
    offset_x = 0
    for i, image in enumerate(images):
        merged[:heights[i], offset_x:offset_x+widths[i]] = image
        offset_x += widths[i]
    return merged


# Predict the value of a given digit.
def predict(digit):
    digit = np.reshape(digit, (1, 28, 28, 1))
    prediction = model.predict(digit)
    prediction = np.argmax(prediction)
    return prediction


# Process the input.
def processInput():
    # Get user input and set up appropriate capture object based on user input.
    print("\nMENU:\n", "[1] Open internal webcam.\n", "[2] Open external webcam.\n", "[3] Open video file.\n")
    user_input = input("Enter an option: ")
    cap = getOption(int(user_input))

    while cap.isOpened():
        # Capture frame-by-frame.
        ret, frame = cap.read()

        # Convert the frame to grayscale.
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection.
        edge_img = cv2.Canny(gray_img, 100, 200)

        # Find the contrours of the digits from the binary image.
        contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours.
        contour_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)

        # Otsu thresholding to obtain a binary image of the digits.
        thresh, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.floodFill(binary_img, None, (0, 0), 0)

        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 5)

        # Stores all the digits found in the frame.
        digit_list = []
        
        for contour in contours:
            # Get (x, y) coordinates, width, and height of the contours.
            x, y, w, h = cv2.boundingRect(contour)

            # Drawing a bounding box around the contour if it's area is between 200 and 8000 pixles.
            if 800 < cv2.contourArea(contour) < 5000 and h < 500:
                cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), (0, 255, 0), 2)

                # Extract digit from the bounding box.
                digit = binary_img[y: y+h, x: x+w]
                gray_digit = gray_img[y: y+h, x: x+w]

                # Appends each digit in the frame to the list, if the hight and width of the bounding box is <= 300.
                if w <= 300 and h <= 300:
                    digit_list.append(gray_digit)

                # Resizing the digit to 28x28 pixles and normalizing it.
                digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                digit = digit / 255.0
                digit = np.array(digit).reshape(-1, 28, 28, 1)[0]
                
                # Predicting the digit.
                digit_pred = predict(digit)

                # Displaying the predicted digit on the bounding box.
                cv2.putText(frame, str(digit_pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Displaying the entire webcam.
        cv2.imshow("Digit Recognizer", frame)
        
        # Displays the individual digits in the frame
        if digit_list != []:
            d_imgs = merge_images(digit_list)
            cv2.imshow("Individual Digits", d_imgs)

        # To quit application.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Execute program.
processInput()

        



