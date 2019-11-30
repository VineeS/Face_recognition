import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("news.jpg")
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray_img, 
scaleFactor= 1.1,
minNeighbors=5 #how many neighbors around the window
)


print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y),(x+w, y+h),(0,255,0),3)

resized = cv2.resize(img,(int(img.shape[1]/3), int(img.shape[0]/3)))


cv2.imshow('Gray', resized)

cv2.waitKey(0)
cv2.distroyAllWindows()