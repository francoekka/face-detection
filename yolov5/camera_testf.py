import cv2

def main():
    # Open the default webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
