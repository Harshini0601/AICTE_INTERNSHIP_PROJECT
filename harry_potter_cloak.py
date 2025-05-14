# import cv2
# import numpy

# def hello(x):
#     print("")

# cap = cv2.VideoCapture(0)
# bars=cv2.namedWindow("bars")

# cv2.createTrackbar("upper_hue","bars",110,180,hello)
# cv2.createTrackbar("upper_saturation","bars",255,255,hello)
# cv2.createTrackbar("upper_value","bars",255,255,hello)
# cv2.createTrackbar("lower_hue","bars",68,180,hello)
# cv2.createTrackbar("lower_saturation","bars",55,255,hello)
# cv2.createTrackbar("lower_value","bars",54,255,hello)

# while(True):
#     cv2.waitKey(1000)
#     ret.init, frame = cap.read()
#     if(ret):
#         break

# while(True):
#     ret.frame=cap.read()
#     inspect=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     upper_hue=cv2.getTrackbarPos("upper_hue","bars")
#     upper_saturation=cv2.getTrackbarPos("upper_saturation","bars")
#     upper_value=cv2.getTrackbarPos("upper_value","bars")
#     lower_hue=cv2.getTrackbarPos("lower_hue","bars")
#     lower_saturation=cv2.getTrackbarPos("lower_saturation","bars")
#     lower_value=cv2.getTrackbarPos("lower_value","bars")

#     kernel=numpy.ones((3,3),numpy.uint8)
#     upper_hsv=numpy.array([upper_hue,upper_saturation,upper_value])
#     lower_hsv=numpy.array([lower_hue,lower_saturation,lower_value])

#     mask=cv2.inRange(inspect,lower_hsv,upper_hsv)
#     mask=cv2.medianBlur(mask,3)
#     mask_inv=255.mask
#     mask=cv2.dilate(mask,kernel,5)

#     b=frame[:,:,0]
#     g=frame[:,:,1]
#     r=frame[:,:,2]
#     b=cv2.bitwise_and(mask_inv,b)
#     g=cv2.bitwise_and(mask_inv,g)
#     r=cv2.bitwise_and(mask_inv,r)
#     frame=cv2.merge((b,g,r))\
    
#     b=init_frame[:,:,0]
#     g=init_frame[:,:,1]
#     r=init_frame[:,:,2]
#     b=cv2.bitwise_and(b,mask)
#     g=cv2.bitwise_and(g,mask)
#     r=cv2.bitwise_and(r,mask)
#     frame=cv2.merge((b,g,r))

#     final=imshow("Harry's Cloak",final)
#     cv2.imshow("original",frame)

#     if(cv2.waitkey(3)==ord('q')):
#         break

# cv2.destroyAllWindows()
# cap.release()
import cv2
import numpy as np
import time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    # Instructions
    print("\n=== Harry Potter's Invisibility Cloak ===")
    print("1. Make sure you're not in the frame")
    print("2. Wait for background capture (3 seconds)")
    print("3. Put on your blue colored cloth/cloak")
    print("4. Press 'q' key or Ctrl+C to quit\n")

    print("Capturing background in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Capture multiple background frames and average them for stability
    background_frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            background_frames.append(frame)
    
    if not background_frames:
        print("Error: Could not capture background")
        return
    
    # Average the background frames
    background = np.average(background_frames, axis=0).astype(np.uint8)

    print("\nBackground captured! You can enter the frame with your blue cloak now.")
    print("To exit the program, press 'q' key while focused on any window or press Ctrl+C in terminal")

    # Enhanced HSV ranges for better blue detection
    # Define two ranges of blue to catch both light and dark blues
    lower_blue1 = np.array([95, 70, 50])   # Light blue lower bound
    upper_blue1 = np.array([135, 255, 255]) # Light blue upper bound
    lower_blue2 = np.array([85, 50, 50])    # Dark blue lower bound
    upper_blue2 = np.array([95, 255, 255])  # Dark blue upper bound

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Convert frame to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create masks for both ranges of blue
            mask1 = cv2.inRange(hsv_frame, lower_blue1, upper_blue1)
            mask2 = cv2.inRange(hsv_frame, lower_blue2, upper_blue2)
            
            # Combine the masks
            mask = cv2.bitwise_or(mask1, mask2)

            # Enhanced mask processing
            # 1. Remove noise
            mask = cv2.medianBlur(mask, 5)
            
            # 2. Dilate to fill gaps
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # 3. Erode to remove small detections
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # 4. Apply Gaussian blur to smooth edges
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            
            # 5. Apply threshold to make mask binary again
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Create inverse mask
            mask_inv = cv2.bitwise_not(mask)

            # Apply the invisibility effect
            # Use float32 for better color blending
            background_visible = cv2.bitwise_and(background, background, mask=mask).astype(np.float32)
            current_frame_visible = cv2.bitwise_and(frame, frame, mask=mask_inv).astype(np.float32)
            
            # Blend the images with weighted addition for smoother transition
            alpha = 0.7  # Adjust this value between 0 and 1 to control the blend
            final_output = cv2.addWeighted(background_visible, alpha, current_frame_visible, 1.0, 0)
            final_output = final_output.astype(np.uint8)

            # Show results
            cv2.imshow("Original", frame)
            cv2.imshow("Invisible Cloak Effect", final_output)
            
            # Optional: Show mask for debugging
            cv2.imshow("Mask", mask)

            # Break loop on 'q' press (check more frequently)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting program...")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Program ended successfully!")

if __name__ == "__main__":
    main()


