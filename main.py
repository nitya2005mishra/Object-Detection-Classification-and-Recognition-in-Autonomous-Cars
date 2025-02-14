import cv2
import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import torchvision.transforms as T

# Load the pre-trained SSD model with weights
model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
model.eval()

# COCO object labels
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 
    13: 'dog', 14: 'horse', 15: 'sheep', 16: 'cow', 17: 'elephant', 18: 'bear', 
    19: 'zebra', 20: 'giraffe', 21: 'backpack', 22: 'umbrella', 23: 'handbag', 
    24: 'tie', 25: 'suitcase', 27: 'frisbee', 28: 'skis', 29: 'snowboard', 
    30: 'sports ball', 31: 'kite', 32: 'baseball bat', 33: 'baseball glove', 
    34: 'skateboard', 35: 'surfboard', 36: 'tennis racket', 37: 'bottle', 
    38: 'wine glass', 39: 'cup', 40: 'fork', 41: 'knife', 42: 'spoon', 43: 'bowl', 
    44: 'banana', 45: 'apple', 46: 'sandwich', 47: 'orange', 48: 'broccoli', 
    49: 'carrot', 50: 'hot dog', 51: 'pizza', 52: 'donut', 53: 'cake', 54: 'chair', 
    55: 'couch', 56: 'potted plant', 57: 'bed', 58: 'dining table', 59: 'toilet', 
    60: 'tv', 61: 'laptop', 62: 'mouse', 63: 'remote', 64: 'keyboard', 65: 'cell phone', 
    67: 'microwave', 68: 'oven', 69: 'toaster', 70: 'sink', 71: 'refrigerator', 
    72: 'book', 73: 'clock', 74: 'vase', 75: 'scissors', 76: 'teddy bear', 
    77: 'hair drier', 78: 'toothbrush'
}

# Start the webcam capture
cap_webcam = cv2.VideoCapture(0)  # 0 for default webcam

# Open the video file for processing
video_file_path = r'C:\Users\HP\Downloads\Real-time-Object-Detection-and-Classification-using-SSD-Algorithm-master\Object Detection, Classification and Recognition using SSD Model\Input_video.mp4'
cap_video = cv2.VideoCapture(video_file_path)

# Check if webcam opened successfully
if not cap_webcam.isOpened() or not cap_video.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Function to preprocess the frame
def transform_frame(frame):
    transform = T.Compose([T.ToTensor()])
    return transform(frame).unsqueeze(0)  # Add batch dimension

while cap_webcam.isOpened() and cap_video.isOpened():
    ret_webcam, frame_webcam = cap_webcam.read()
    ret_video, frame_video = cap_video.read()

    if not ret_webcam or not ret_video:
        break  # Exit the loop if no frames are left from either capture

    # Process webcam frame
    frame_webcam_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
    input_tensor_webcam = transform_frame(frame_webcam_rgb)
    with torch.no_grad():
        prediction_webcam = model(input_tensor_webcam)

    boxes_webcam = prediction_webcam[0]['boxes'].cpu().numpy()
    labels_webcam = prediction_webcam[0]['labels'].cpu().numpy()
    scores_webcam = prediction_webcam[0]['scores'].cpu().numpy()

    # Process video file frame
    frame_video_rgb = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
    input_tensor_video = transform_frame(frame_video_rgb)
    with torch.no_grad():
        prediction_video = model(input_tensor_video)

    boxes_video = prediction_video[0]['boxes'].cpu().numpy()
    labels_video = prediction_video[0]['labels'].cpu().numpy()
    scores_video = prediction_video[0]['scores'].cpu().numpy()

    # Threshold for displaying detections
    threshold = 0.5

    # Draw bounding boxes and labels on the webcam frame
    for i in range(len(boxes_webcam)):
        if scores_webcam[i] > threshold:
            box = boxes_webcam[i]
            label = labels_webcam[i]
            score = scores_webcam[i]
            label_name = COCO_LABELS.get(label, "Unknown")
            cv2.rectangle(frame_webcam, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame_webcam, f'{label_name}: {score:.2f}', 
                        (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw bounding boxes and labels on the video file frame
    for i in range(len(boxes_video)):
        if scores_video[i] > threshold:
            box = boxes_video[i]
            label = labels_video[i]
            score = scores_video[i]
            label_name = COCO_LABELS.get(label, "Unknown")
            cv2.rectangle(frame_video, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame_video, f'{label_name}: {score:.2f}', 
                        (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the processed frames
    cv2.imshow('Webcam Object Detection', frame_webcam)
    cv2.imshow('Video File Object Detection', frame_video)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects
cap_webcam.release()
cap_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
