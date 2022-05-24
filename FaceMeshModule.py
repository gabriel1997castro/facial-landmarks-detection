import cv2
import mediapipe as mp
import time

lipsInner = {
78: True,
191: True,
80: True,
81: True,
82: True,
13: True,
312: True,
311: True,
310: True,
415: True,
308: True,
324: True,
318: True,
402: True,
317: True,
14: True,
87: True,
178: True,
88: True,
95: True,
78: True
}

lipsOutter = {
61: True,
185: True,
40: True,
39: True,
37: True,
0: True,
267: True,
269: True,
270: True,
409: True,
291: True,
375: True,
321: True,
405: True,
314: True,
17: True,
84: True,
181: True,
91: True,
146: True,
61: True
}

class FaceMeshDetector():

    def __init__(self, staticMode = False, maxFaces= 2, minDetectionCon=0.5, minTrackcon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackcon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMash(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        if self.results.multi_face_landmarks:
            faces = []
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS)

                face = []
                # for id, lm in enumerate(faceLms.landmark):
                #     # print(lm)
                #     ih, iw, ic = img.shape
                #     x, y = int(lm.x*iw), int(lm.y*ih)
                #     if lipsOutter.get(id):
                #         cv2.putText(img, f'{int(id)}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                #
                #     face.append([x, y])
                faces.append(face)
            return img, faces

def main():
    VIDEO_WIDTH = 3
    VIDEO_HEIGHT = 4
    VIDEO_BRIGHTNESS = 10

    cap = cv2.VideoCapture(0)
    cap.set(VIDEO_WIDTH, 640)
    cap.set(VIDEO_HEIGHT, 480)
    cap.set(VIDEO_BRIGHTNESS, 100)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMash(img, True)
        if len(faces) != 0:
            print(len(faces))
        cTime = time.perf_counter()
        fps = 1.0 / (cTime - pTime)  # current - previous
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()