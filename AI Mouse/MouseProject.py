import cv2
import numpy as np
import HandTrackingmodule as htm
import time
import autopy

# Width and Height
wCam, hCam = 640, 480
frameR = 100 # Frame reduction
smoothening = 5

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Capture video
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # Find hand label marks
    success, img = cap.read()

    # Check if the frame is read successfully
    if not success:
        print("Error: Failed to capture frame")
        break

    # Convert frame to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index, middle, and ring finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger
        x3, y3 = lmList[16][1:]  # Ring finger

        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2) # OUTLINE BOX

        # Only index finger Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Both index and middle finger are up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

                # Index, middle, and ring finger are up: right-click mode
                if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    # Right-click action
                    autopy.mouse.toggle(autopy.mouse.Button.RIGHT, True)
                    autopy.mouse.toggle(autopy.mouse.Button.RIGHT, False)

        # Frame rate
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
