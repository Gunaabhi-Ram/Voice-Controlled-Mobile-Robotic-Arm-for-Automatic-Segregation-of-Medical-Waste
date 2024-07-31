import torch
import numpy as np
import cv2
import pafy
import time
from dronekit import connect, VehicleMode
import tkinter as tk
import speech_recognition as sr

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    global cap
    url = "http://192.168.136.17:5000/"
    cap = cv2.VideoCapture(url)


    def __init__(self,model_path):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

    def load_model(self,model_path):
        model_path = "C:/Users/chotu/Desktop/ps/best_7.pt"
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        area = 0  # Initialize area variable
        center = None  # Initialize center point variable

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            
                if self.class_to_label(labels[i]) == 'tablets':
                    l = x2 - x1
                    b = y2 - y1
                    area = l * b  # Calculate area for the 'person' class
                    center = (x1 + l // 2, y1 + b // 2)  # Calculate the center point of the bounding box

                    # Display the center of the bounding box
                    cv2.circle(frame, center, 3, (0, 0, 255), -1)

        return frame, area, center

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,+
        and write the output into a new file.
        :return: void
        """

        while cap.isOpened():     
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame, area, center = self.plot_boxes(results, frame)  # Receive the area value and center point
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Calculate the center of the screen
            height, width, _ = frame.shape
            center_x = width // 2
            center_y = height // 2
            cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # print(center_x, center_y)

            cv2.imshow("img", frame)

            if area != 0:  # Check if an area value is available
                return area, center

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        

def forward():
   vehicle.channels.overrides['3'] = 1200

def backward():
  vehicle.channels.overrides['3'] = 1800
  time.sleep(1)
  vehicle.channels.overrides['3'] = 1500
  vehicle.channels.overrides['1'] = 1500


# Turn right
def right():
  vehicle.channels.overrides['3'] = 1500
  vehicle.channels.overrides['1'] = 1800

#left
def left():
  vehicle.channels.overrides['3'] = 1500
  vehicle.channels.overrides['1'] = 1200
  

def move():
  while True:
    areas , center = detection()
    if center[0] < 320:
      left()
      vehicle.channels.overrides['3'] = 1500
      vehicle.channels.overrides['1'] = 1500
    elif center[0] > 320:
      right()
      vehicle.channels.overrides['3'] = 1500
      vehicle.channels.overrides['1'] = 1500
    else:
      forward()
      if areas <= 200000:
        vehicle.channels.overrides['3'] = 1500
        vehicle.channels.overrides['1'] = 1500
        break


def key(event):
    if event.char == event.keysym:  # Standard keys
        if event.keysym == 'q':
            print("QUIT")
            vehicle.armed = False
            vehicle.close()
            exit()
        elif event.keysym == 'r':
            # Add RTL mode implementation
            pass
    else:  # Non-standard keys
        if event.keysym == 'Up':
            forward()
        elif event.keysym == 'Down':
            backward()
        elif event.keysym == 'Right': 
            right()
        elif event.keysym == 'Left':
            left()
        elif event.keysym == 'space': 
            print('hi')
            recognize_live_voice_command()


def recognize_live_voice_command():
    # Create a recognizer object
    r = sr.Recognizer()
    
    # with sr.Microphone(device_index=1) as source:
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")

        # Continuously listen for voice commands in real-time
        while True:
            try:
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source)

                # Listen for the user's voice command
                audio = r.listen(source)
                # Recognize speech using Google Speech Recognition
                command = r.recognize_google(audio)
                print("Voice command:", command)

                # Process the recognized command
                process_voice_command(command)
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

def process_voice_command(command):
    if 'forward' in command:
        forward()
    elif 'backward' in command:
        backward()
    elif 'right' in command:
        right()
    elif 'left' in command:
        left()
    # Add more voice commands and corresponding actions as needed
    elif 'move' in command:
        move()
    else:
        print("Unrecognized voice command")



# Create a new object and execute.
detection = ObjectDetection("C:/Users/chotu/Desktop/ps/best_7.pt")
area, center = detection()

connection_string = 'COM9' # Replace with the serial port of your Pixhawk
baud_rate = 57600 # Replace with the baud rate that your Pixhawk is configured to use
vehicle = connect(connection_string, baud=baud_rate, wait_ready=False)

# Set the mode of the vehicle to MANUAL
vehicle.mode = VehicleMode("MANUAL")

# Arm the vehicle
vehicle.armed = True



root = tk.Tk()
print(">> Control the drone with the arrow keys. Press r for RTL mode")
root.bind_all('<Key>', key)
root.mainloop()


