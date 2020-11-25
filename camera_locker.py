import cv2
import numpy as np
import os 
#socket
import socket
import threading

#GPIO------------------------------------------------------------------------------------------------------------------
import RPi.GPIO as GPIO
import time
OFFSE_DUTY = 0.5        #define pulse offset of servo
SERVO_MIN_DUTY = 2.5+OFFSE_DUTY     #define pulse duty cycle for minimum angle of servo
SERVO_MAX_DUTY = 12.5+OFFSE_DUTY    #define pulse duty cycle for maximum angle of servo
servoPin = 12
buttonPin = 11    # define buttonPin

cell_phone_count = 0

def map( value, fromLow, fromHigh, toLow, toHigh):  # map a value from one range to another range
    return (toHigh-toLow)*(value-fromLow) / (fromHigh-fromLow) + toLow

def setup():
    global p
    GPIO.setmode(GPIO.BOARD)         # use PHYSICAL GPIO Numbering
    GPIO.setup(servoPin, GPIO.OUT)   # Set servoPin to OUTPUT mode
    GPIO.output(servoPin, GPIO.LOW)  # Make servoPin output LOW level

    p = GPIO.PWM(servoPin, 50)     # set Frequece to 50Hz
    p.start(0)                     # Set initial Duty Cycle to 0
    GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)    # set buttonPin to PULL UP INPUT mode    

def servoWrite(angle):      # make the servo rotate to specific angle, 0-180 
    if(angle<0):
        angle = 0
    elif(angle > 35):
        angle = 35
    p.ChangeDutyCycle(map(angle,0,180,SERVO_MIN_DUTY,SERVO_MAX_DUTY)) # map the angle to duty cycle and output it

def destroy():
    p.stop()
    GPIO.cleanup()

def loop():
    for dc in range(0, 36, 1):   # make servo rotate from 0 to 180 deg
        servoWrite(dc)     # Write dc value to servo
        time.sleep(0.001)
    time.sleep(5.0)
    for dc in range(35, -1, -1): # make servo rotate from 180 to 0 deg
        servoWrite(dc)
        time.sleep(0.001)
    time.sleep(1.0)

def call_cellphone():
    global cell_phone_count
    while True:
        coon, addr = server.accept()
        clientMessage = str(coon.recv(1024), encoding='utf-8')

        print('Client message is:', clientMessage)

        serverMessage = 'I\'m here'
        coon.sendall(serverMessage.encode())
        coon.close()
        cell_phone_count = 1
        time.sleep(5.0)
        cell_phone_count = 0

#GPIO------------------------------------------------------------------------------------------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Marcelo', 'Kevin', 'Ilza', 'Z', 'W'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

buttonflag = 1
face_control = 0
setup()

#service setting
HOST = '192.168.0.26'
POST = 8000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, POST))
server.listen(10)

thread = threading.Thread(target=call_cellphone)
thread.start()

while True:
    if(buttonflag == 0):
        setup()
        buttonflag = 1
    if (GPIO.input(buttonPin)==GPIO.LOW or cell_phone_count == 1): # if button is pressed
        loop()
        destroy()
        buttonflag = 0

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            print(id)
            print(confidence)
#GPIO----------------------------------------------------------------------------------
            if (id == "Kevin"):
                face_control +=1
                if(face_control == 10):
                    print ('open door')
                    face_control = 0
                    loop()
                    destroy()
                    buttonflag = 0
                    face_control =0
                
#GPIO----------------------------------------------------------------------------------
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
