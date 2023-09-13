import streamlit as st
import cv2
import numpy as np
import tempfile

def main():
    st.set_page_config(page_title="Eulerian Magnification", page_icon=":eyeglasses:")
    tab1, tab2 = st.tabs(["Live Feed", "Upload Video"])
    tab1.write("Eulerian Magnification for Live Feed")
    tab2.write("Eulerian Magnification for Uploaded Video")
    
    # You can also use "with" notation:
    with tab1:
        st.title("Webcam Display Steamlit App")
        st.caption("Powered by OpenCV, Streamlit")
        network = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        start_button_pressed = st.button("Start")
        stop_button_pressed = st.button("Stop")
        while cap.isOpened() and not stop_button_pressed and start_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break
            frame = cv2.resize(frame, (500, 350))

            # Grap dimensions to blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Pass blot through network to perform facial detection
            network.setInput(blob)
            detections = network.forward()
            count = 0
            
            for i in range(0, detections.shape[2]):
                # Extract confidence assoficated with predictions.
                confidence = detections[0, 0, i, 2]

                # Filter based on confidence
                if confidence < 0.5:
                    continue
                count += 1

                # compute BBOX
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw box
                text = "{:.2f}%".format(confidence * 100) + \
                    ", Count: " + str(count)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY),
                            (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
        cap.release()
        cv2.destroyAllWindows()

    with tab2:
        f = st.file_uploader('Upload a video file', type=['mp4'])
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            cap = cv2.VideoCapture(tfile.name)
            frame_placeholder = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Video Capture Ended")
                    break
                frame = cv2.resize(frame, (500, 350))

                # Grap dimensions to blob
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(
                    frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                # Pass blot through network to perform facial detection
                network.setInput(blob)
                detections = network.forward()
                count = 0
                
                for i in range(0, detections.shape[2]):
                    # Extract confidence assoficated with predictions.
                    confidence = detections[0, 0, i, 2]

                    # Filter based on confidence
                    if confidence < 0.5:
                        continue
                    count += 1

                    # compute BBOX
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw box
                    text = "{:.2f}%".format(confidence * 100) + \
                        ", Count: " + str(count)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY),
                                (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame,channels="RGB")
                if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                    break
            cap.release()
            cv2.destroyAllWindows()
if __name__ == "__main__":
    main()