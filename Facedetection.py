import cv2
import mediapipe as mp
import argparse


def process_img(img, face_detection):

    H, W,_ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height


            x1= int(x1 * W)
            y1= int(y1 * H)
            w= int(w * W)
            h= int(h * H)

            img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)

    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default= None)

args = args.parse_args()

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1) as face_detection:
#model_selection=0 for short-range model(<2m)
#model_selection=1 for full-range model(<5m)

    if args.mode in ["image"]:

        
        img = cv2.imread(args.filePath)

        

        img = process_img(img, face_detection)

        cv2.imshow('img',img)
        cv2.waitKey(0)

    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        while ret:
            img = process_img(img,face_detection)
            
            ret, frame = cap.read()


        cap.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while ret:
            frame = process_img(frame,face_detection)
            
            cv2.imshow('frame', frame)
            cv2.waitKey(25)

            ret, frame = cap.read()



        cap.release()



