import streamlit as st
import pandas as pd
import numpy as np
import cv2
import plotly.express as px  # interactive charts
import tempfile
import sys
sys.path.append("../Eulerian_Magnification/")
sys.path.append("../validate_models/")
from Methodized_Eulerian import EulerianMagnification
from Multiface_Eulerian import MultifaceEulerianMagnification
from main import *

def video_upload(video_data=None, method="Traditional Eulerian Magnification"):
    st.title("Video upload")
    
    placeholder = st.empty()
    if method == "Traditional Eulerian Magnification":
        eulerian_processor = EulerianMagnification()
    elif method == "Custom implemented Eulerian Magnification":
        eulerian_processor = MultifaceEulerianMagnification()
    elif method == "MTTS":
        pass
    elif method == "HR_CNN":
        pass
    else:
        print("No method selected")
        st.write("No method selected")
        return
    bpms = []
    bpms_second = [0]
    nans = []
    bpmES_deep = []

    time_idx = 1
    multiface_df = pd.DataFrame(columns=["Time", "BPM", "person"])

    if not video_data:
        st.write("Upload a file before continuing")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_data.read())
        test = Test_Methods(videoFileName = tfile.name)
        if method == "MTTS":
            with st.spinner(f'Computing HR using {method}'):
                bpmES_deep = test.test_deep(method = "MTTS_CAN")
        elif method == "HR_CNN":
            with st.spinner(f'Computing HR using {method}'):
                bpmES_deep = test.test_deep(method = "HR_CNN")
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended")
                break
    
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if method == "Traditional Eulerian Magnification":
                eulerian_processor.process_frame(frame, display_pyramid=True)
                bpmES = eulerian_processor.get_bpm_over_time()
                bpms.append(bpmES.mean())
            elif method == "Custom implemented Eulerian Magnification":
                eulerian_processor.process_frame_streamlit(frame)
                bpmES = eulerian_processor.get_bpm_over_time()
                temp_df = pd.DataFrame.from_dict(bpmES).mean()
                temp_df = temp_df.reset_index()
                temp_df.columns = ["person", "BPM"]
                temp_df["Frame"] = time_idx
                time_idx += 1
                multiface_df = pd.concat([multiface_df, temp_df])
                # multiface_df.groupby(multiface_df.index // 30).mean()

                # if len(bpms) > 200:
                #   bpms = bpms[-200:]
                # get hr for each second
            if len(bpms) > 30:
                if method == "MTTS" or method == "HR-CNN":
                    continue
                bpm = np.mean(bpms[-30:])
                bpms_second.append(bpm)
                # new_row = pd.DataFrame([[time_idx, bpm, "Eulerian"]], columns=["Time", "BPM", "type"])
                # plot_df = pd.concat([plot_df, new_row])
                time_idx += 1
                bpms = []

            with placeholder.container():
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown("### BPM over time")
                    st.image(frame, channels="RGB")
                with fig_col2:
                    st.markdown("### BPM Statistics")
                    if method == "Traditional Eulerian Magnification":
                        bpm_df = pd.DataFrame(bpms_second, columns=["BPM_Eulerian"])
                        bpm_df = bpm_df.describe()
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(
                            index={"mean": "Mean", "max": "Max", "std": "Std"}
                        )
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                    elif method == "Custom implemented Eulerian Magnification":
                        # describe bpm for each person
                        bpm_df = multiface_df.copy()
                        bpm_df.Frame = bpm_df.Frame.astype(int)
                        bpm_df = bpm_df.pivot(
                            index="Frame", columns="person", values="BPM"
                        )
                        bpm_df = bpm_df.reset_index().drop("Frame", axis=1)
                        bpm_df = bpm_df.rename_axis("person", axis=1)
                        bpm_df = bpm_df.astype(float)
                        bpm_df = bpm_df.describe()
                        # print(bpm_df)
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(
                            index={"mean": "Mean", "max": "Max", "std": "Std"}
                        )
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                    elif method == "MTTS" or method == "HR_CNN":
                        bpm_df = pd.DataFrame(bpmES_deep, columns=[method])
                        bpm_df = bpm_df.describe()
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(
                            index={"mean": "Mean", "max": "Max", "std": "Std"}
                        )
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                    if method == "Custom implemented Eulerian Magnification":
                        st.write(bpm_df.iloc[1:, :])
                    else:
                        st.write(bpm_df)

                fig = px.line(
                    x=np.arange(len(bpms_second)),
                    y=bpms_second,
                    labels={"x": "Time", "y": "BPM"},
                )
                if method == "Custom implemented Eulerian Magnification":
                    fig = px.line(
                        multiface_df,
                        x="Frame",
                        y="BPM",
                        color="person",
                        labels={"x": "Frame", "y": "BPM"},
                    )
                if method == "MTTS" or method == "HR_CNN":
                    nans = [np.nan] * 6
                    bpms = nans + bpmES_deep
                    data = np.array([range(1, len(bpms)+1), bpms]).T
                    
                    plot_df = pd.DataFrame(data=data, columns=["Time", "BPM"])
                    fig = px.line(plot_df[plot_df["Time"].astype(int) > 6], x="Time", y="BPM",
                                        labels={"x": "Time", "y": "BPM"})
                st.write(fig)
