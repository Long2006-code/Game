import cv2
import mediapipe as mp
import pydirectinput

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x and \
                   hand_landmarks.landmark[19].x > hand_landmarks.landmark[20].x:
              pydirectinput.keyDown('down')
              pydirectinput.keyUp('down')
          elif hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
              pydirectinput.keyDown('left')
              pydirectinput.keyUp('left')
          elif hand_landmarks.landmark[19].x > hand_landmarks.landmark[20].x:
              pydirectinput.keyDown('right')
              pydirectinput.keyUp('right')
          elif hand_landmarks.landmark[7].y > hand_landmarks.landmark[8].y:
              pydirectinput.keyDown('up')
              pydirectinput.keyUp('up')
          mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()