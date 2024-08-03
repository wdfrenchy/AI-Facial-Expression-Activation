import pygame
import time
import cv2
import dlib
import numpy as np

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
elmo = cv2.imread('/Users/frenwd24/Desktop/AIPython/elmo.gif', -1)

cap = cv2.VideoCapture(0)

def play_mp3(path_to_mp3):
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load the MP3 music file
    pygame.mixer.music.load(path_to_mp3)
    
    # Start playing the music
    pygame.mixer.music.play()
    
    # Wait for the music to play before exiting, or use pygame.mixer.music.get_busy() to check if still playing
    while pygame.mixer.music.get_busy():
        # Keep the script running while the music is playing
        pygame.time.Clock().tick(10)

# def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
#     overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
#     h, w, _ = overlay.shape  # Size of foreground
#     rows, cols, _ = src.shape  # Size of background Image
#     y, x = pos[0], pos[1]  # Position of foreground/overlay image

#     for i in range(h):
#         for j in range(w):
#             if x + i >= rows or y + j >= cols:
#                 continue
#             alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
#             src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
#     return src

taken = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

    for face in faces:
        landmarks = predictor(gray, face)

        left_corner = 48
        #48
        left_lip = 60
        #60
        right_corner = 54
        #54
        right_lip = 64
        #64

        # Loop through each landmark point around the mouth
        for n in range(1, 68):  # Dlib's mouth landmarks are from point 48 to 67
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(blank_image, (x, y), 2, (255, 0, 0), -1)

        cv2.circle(blank_image, (landmarks.part(left_corner).x,landmarks.part(left_corner).y), 2, (0, 0, 255), -1)    
        cv2.circle(blank_image, (landmarks.part(left_lip).x,landmarks.part(left_lip).y), 2, (0, 255, 0), -1)    
        cv2.circle(blank_image, (landmarks.part(right_corner).x,landmarks.part(right_corner).y), 2, (0, 0, 255), -1)    
        cv2.circle(blank_image, (landmarks.part(right_lip).x,landmarks.part(right_lip).y), 2, (0, 255, 0), -1)  

        smiling = landmarks.part(left_corner).y < landmarks.part(left_lip).y and landmarks.part(right_corner).y < landmarks.part(right_lip).y
        print(smiling)
        if smiling and not taken:
            cv2.putText(blank_image, "smiling", (landmarks.part(right_corner).x + 10, landmarks.part(right_corner).y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
            cv2.putText(frame, "smiling", (landmarks.part(right_corner).x + 10, landmarks.part(right_corner).y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),2)
            cv2.imwrite('smiling.png', frame)
            #transparentOverlay(blank_image, elmo, pos=(x, y), scale = 1)
            play_mp3('/Users/frenwd24/Desktop/AIPython/camera-shutter-6305.mp3')
            #play_mp3('/Users/frenwd24/Desktop/AIPython/happy.mp3')
            taken = True
        
        if not smiling:
            taken = False
            #pygame.mixer.music.stop()

    cv2.imshow("Frame", blank_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
