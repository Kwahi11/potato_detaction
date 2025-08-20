#coding:utf-8
import cv2
from ultralytics import YOLO
import numpy as np
# 所需加载的模型目录
path = 'models/best.pt'

# Load the YOLOv8 model
model = YOLO(path)
#0是电脑自带摄像头，1是外接摄像头
ID = 0


while ID < 10:
    cap = cv2.VideoCapture(ID)
    # get a frame
    ret, frame = cap.read()
    if ret == False:
        ID += 1
    else:
        print('摄像头ID:', ID)
        break


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Assume frame is a color frame
        color_image = frame
        # For simplicity, assume depth_image is a placeholder
        depth_image = np.zeros_like(color_image)

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Assume x1, x2, y1, y2 are obtained from the detection results
        # Here we need to get them from results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)

                # Scale the bounding box coordinates to the resolution of the depth image
                scale_x = depth_image.shape[1] / color_image.shape[1]
                scale_y = depth_image.shape[0] / color_image.shape[0]
                center_x = int((x1 + x2) / 2 * scale_x)
                center_y = int((y1 + y2) / 2 * scale_y)

                # Ensure the coordinates are within the depth image range
                if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                    depth_value = depth_image[center_y, center_x]
                    print(f"Object detected at (x, y, depth): ({center_x}, {center_y}, {depth_value})")
                else:
                    print(f"Object detected at (x, y) but out of depth image bounds: ({center_x}, {center_y})")


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()