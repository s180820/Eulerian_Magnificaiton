import cv2
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
from MethodizedEulerian import EulerianMagnification
from multiface import MultifaceEulerianMagnification

def demo():
    st.title("Validation Demo")
    st.write('''This demo is to show the Eulerian Video Magnification algorithm in action. It uses 
            a video from the rPPG bench dataset and methods from pyVHR to calculate the heart rate.''')
    placeholder2 = st.empty()
    cap2 = cv2.VideoCapture("Validation/val_vid.mp4")
    eulerian_processor = EulerianMagnification()
    multiface_processor = MultifaceEulerianMagnification()
    bpms=[]
    bpms_second = [0]
    rr = pd.read_csv('Validation/P1LC1_Mobi_RR-intervals.rr', header=None)
    #split data column into two columns
    rr = rr[0].str.split(' ', expand=True)
    rr = rr.rename(columns={0:'time', 1:'rr'})
    hr = 60/rr['rr'].astype(float)
    # repeat "Ground Truth" for 60 times
    gt = np.repeat("Ground Truth", 60)

    # create dataframe for plot 
    data = np.array([range(1, 61), hr[:60].values, gt]).T
    #plot_df = pd.DataFrame([*range(1, 61), hr[:60].values, gt], columns=["Time", "BPM", "type"])
    plot_df = pd.DataFrame(data = data, columns=["Time", "BPM", "type"])
    
    # add HR_CNN and MTTS_CAN data
    hr_cnn = np.load("../validate_models/bpmES_HR_CNN.npy")
    mtts_can = np.load("../validate_models/bpmES_MTTS_CAN.npy")
    #make df for HR_CNN
    hr_cnn_df = pd.DataFrame([range(7, 61), hr_cnn, np.repeat("HR_CNN", 54)]).T
    hr_cnn_df.columns = ["Time", "BPM", "type"]
    #make df for MTTS_CAN
    mtts_can_df = pd.DataFrame([range(7, 61), mtts_can, np.repeat("MTTS_CAN", 54)]).T
    mtts_can_df.columns = ["Time", "BPM", "type"]
    #concatenate all dataframes
    plot_df = pd.concat([plot_df, hr_cnn_df, mtts_can_df])
    plot_df = plot_df.reset_index(drop=True)

    #make stats df
    stats_df = plot_df.copy()
    stats_df.Time = stats_df.Time.astype(int)
    #make a column for each type with bpm values
    stats_df = stats_df.pivot(index="Time", columns="type", values="BPM")
    stats_df = stats_df.reset_index().drop("Time", axis=1)
    stats_df = stats_df.rename_axis(None, axis=1)
    stats_df["Ground Truth"] = stats_df["Ground Truth"].astype(float)
    stats_df["HR_CNN"] = stats_df["HR_CNN"].astype(float)
    stats_df["MTTS_CAN"] = stats_df["MTTS_CAN"].astype(float)
    
    stats_df_des = stats_df.describe()
    stats_df_des = stats_df_des.drop(["count", "min", "25%", "50%", "75%"])
    stats_df_des = stats_df_des.rename(index={"mean": "Mean", "max": "Max", "std": "Std"})
    stats_df_des = stats_df_des.T
    stats_df_des_org = stats_df_des.round(2)

    time_idx = 1

    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            st.write("The video capture has ended")
            break
        #img = frame.to_ndarray(format="bgr24")
        eulerian_processor.process_frame(frame)
        bpmES = eulerian_processor.get_bpm_over_time()
        

        bpms.append(bpmES.mean())
            #if len(bpms) > 200:
                #   bpms = bpms[-200:]
            # get hr for each second
        if len(bpms) > 30:
            bpm = np.mean(bpms[-30:])
            bpms_second.append(bpm)
            new_row = pd.DataFrame([[time_idx, bpm, "Eulerian"]], columns=["Time", "BPM", "type"])
            plot_df = pd.concat([plot_df, new_row])
            time_idx += 1
            bpms = []
        
        with placeholder2.container():
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.markdown("### BPM over time")
                st.image(frame, channels="RGB")
            with fig_col2:
                st.markdown("### BPM Statistics")
                stats_df_des = stats_df_des_org.copy()
                bpm_df = pd.DataFrame(bpms_second, columns=["BPM_Eulerian"])
                bpm_df = bpm_df.describe()
                bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                bpm_df = bpm_df.rename(index={"mean": "Mean", "max": "Max", "std": "Std"})
                bpm_df = bpm_df.T
                bpm_df = bpm_df.round(2)

                stats_df_des = pd.concat([bpm_df, stats_df_des])
                stats_df_des["MAE"] = [np.abs(stats_df["Ground Truth"] - plot_df[plot_df["type"] == "Eulerian"]["BPM"]).mean(), 
                                           np.abs(stats_df["Ground Truth"] - stats_df["Ground Truth"]).mean(),
                                           np.abs(stats_df["Ground Truth"] - stats_df["HR_CNN"]).mean(),
                                           np.abs(stats_df["Ground Truth"] - stats_df["MTTS_CAN"]).mean()]
                st.write(stats_df_des)

            #fig = px.line(x=np.arange(len(bpms)), y=bpms, labels={"x": "Time", "y": "BPM"})
            fig = px.line(plot_df[plot_df["Time"].astype(int) > 6], x="Time", y="BPM", 
                                    color="type", labels={"x": "Time", "y": "BPM"})
            st.write(fig)

    
