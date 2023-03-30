# Handwritten Digit Recognizer
This Handwritten Digit Recognizition application utilizes the MNIST allows users to utilize their webcam or any given video file to recognize handwritten digits in the video. The application will automatically detect digits in the frame, which will be be accompanied by bounding boxes around each digit, along with their predicted digit value above each bounding box. 

## Requirements
- Python 3
- OpenCV
- Tensorflow.keras 
- Numpy
- Tkinter

## To Run
1) Make sure all the requirments are installed properly before running the application.

2) Using either terminal/command prompt, navigate into the project directory.

3) To run the application, run the following command:
>> python app.py

4) You will be prompted with a menu system with three options: 
>> [1] Open internal camera. <br> [2] Open external camera. <br> [3] Open video file.

5) Pick [1] or [2] to utilize either your internal or external webcam, respectively, to recognize digits. 



## Usage

- Pick [1] or [2] to utilize either your internal or external webcam, respectively, to recognize digits. You will need to present your camera with a blank paper which contains a series of handwritten digits.

- Pick [3] to select a video file to recognize digits.
### Note:
- To demo option [3], you may use the provided sample video file, located in the project directory, labeled:
>> vid_sample.mov

## Limitations
To ensure that the application runs smoothly, please make sure to consider the following:
- When using option [1] or [2], use your webcam facing a plain background with minimal background decors, furniture, etc. Failing to do so will result in the application to pick up various contours in the background, making the application perform slow. 
- Be sure to write your numbers in a large, readable format on a BLANK, WHITE piece of paper, so that the application can detect digits more efficiently.

## Credits
- MNIST dataset was used to train the Convolutional Neural Network model, 'model.h5.'
- Full implementation of the CNN model can be found in:
>> mnist_cnn.ipynb
