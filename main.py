import subprocess
import os

import numpy as np
import cv2
from collections import Counter

from keras.preprocessing import image
import speech_recognition as sr
from keras.models import model_from_json

from matplotlib import pyplot as plt
import tkinter as tk

#--------------------------------------Loading Files---------------------------------#

# Enter the abolute video path here (make sure there isnt a corresponding audio file yet)
video_path = 'C:/Users/Jaspreet Singh/Videos/interview.mp4'

# Loading OpenCV Cascade Classifier for Face Detection        
face_cascade = cv2.CascadeClassifier('C:/Users/Jaspreet Singh/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Loading emotion recognition model and weights 
model = model_from_json(open('C:/Users/Jaspreet Singh/Downloads/fer.json', 'r').read())
model.load_weights('C:/Users/Jaspreet Singh/Downloads/fer.h5') #load weights

#----------------------------------Speech Detection Part-----------------------------------#

# Exracting the video base name (Example: interview.mp4)
video_base_name = os.path.basename(video_path)

# Extracting video name
video_name = os.path.basename(video_base_name).split('.')[0]

# Getting video directory
video_dir = os.path.dirname(video_path)

# Creating output (audio) file path
audio_path = video_dir + '/' + video_name + '.wav'

print('Input file: {}'.format(video_path))

# Creating subprocess to convert video file to audio
subprocess.call(['ffmpeg', '-i', video_path, '-codec:a', 'pcm_s16le', '-ac', '1', audio_path])

print('Output file: {}'.format(audio_path))

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

with sr.AudioFile(audio_path) as source:
    audio = r.listen(source)
    try:
        # Google speech recognition (You can select from other options)
        text = r.recognize_google(audio)
        
        # Printing speech
        print('Speech Detected:')
        print(text)
     
    except:
        print('Could not hear anything!')

#-------------------------------------------------------------------------------------------#
        
#------------------------------------Emotion Detection Part---------------------------------#
        
# Capturing the video using from the path
cap = cv2.VideoCapture(video_path)

# Emotion labels
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

# Creating empty list which will be used to store emotions
emotion_list = []

# Reading video frame by frame
while(True):
   
    ret, img = cap.read()
    
    # Reading till the end of the video
    if ret:
        
        # Converting to greyscale
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        # Detecting faces
        faces = face_cascade.detectMultiScale(gray_img, 1.32, 5)
        
        """Show image using imshow function (removing this will only display images where a 
        face was captured thus cutting out on some of the frames) """
        img = cv2.resize(img, (1000, 800))
        
        # For every face detected
        for (x,y,w,h) in faces:
            
            # Drawing a rectangle 
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), thickness = 2) 
       		
            # Cropping the face
            face = gray_img[int(y):int(y+h), int(x):int(x+w)]
            
            # Resizing the cropped face
            face = cv2.resize(face, (48, 48))
            
            # Converted face image to pixels
            img_pixels = image.img_to_array(face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
       		
            # Scalling image
            img_pixels = img_pixels/255
            
            # Using the model to predict the detected face
            predictions = model.predict(img_pixels)
       		
       		# Finding the index with most value
            max_index = np.argmax(predictions[0])
           	
            # Finding corresponding emotion
            emotion = emotions[max_index]
            
            # Storing detected emotions in a list
            emotion_list.append(emotion)
            
            # Writing detected emotions on the rectangle
            cv2.putText(img, emotion, (int(x), int(y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        color = (255,255,255),
                        thickness = 1)
    
              # Showing the frame with detected face
            cv2.imshow('Emotion Recognizer', img)
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Closing OpevCV		
cap.release()
cv2.destroyAllWindows()

# Counting emotions from the generated list
c = Counter(emotion_list)
#print([(i, c[i] / len(emotion_list) * 100.0) for i in c])

#-------------------------------------------------------------------------------------------#

#--------------------------------------Displaying Results-----------------------------------#

# Using Tkinter window to write detected speech
root = tk.Tk()
root.title("Detected Speech")
T = tk.Text(root, height=10, width=80)
T.pack()
T.insert(tk.END, text)

# Visualizing emotions using a Pie Chart
plt.figure(num='Detected Emotions')
plt.pie([float(v) for v in c.values()], labels=[k for k in c], autopct='%1.0f%%')
plt.title("Emotions Detected throughout the Interview")

plt.show()
tk.mainloop()

#-------------------------------------------------------------------------------------------#
