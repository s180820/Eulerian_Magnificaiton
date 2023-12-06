import numpy as np
from methods import HR_CNN_bvp_pred, MTTS_CAN_deep
from utils import extract_raw, get_fps, BVP_windowing, BVP_to_BPM_2

class Test_Methods:
    def __init__(self, videoFileName=None):
        if videoFileName is None:
            self.videoFileName = '/Users/gustavlarsen/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Uni/Advanced Project/pyVHR/val_vid.mp4'
        else:
            self.videoFileName = videoFileName
        self.wsize = 6                  # window size in seconds
        self.fps = get_fps(self.videoFileName)
        self.nFFT = 2048//1


    def test_deep(self, method="HR_CNN", save=False):
        print("Testing {}...".format(method))
        print("extracting frames...")
        frames = extract_raw(self.videoFileName)
        #only choose 60 seconds
        #frames = frames[:60*self.fps]
        print("predicting BVP...")
        if method == "HR_CNN":
            bvp_pred = HR_CNN_bvp_pred(frames)
        elif method == "MTTS_CAN":
            bvp_pred = MTTS_CAN_deep(frames, self.fps, verb=1)
        #bvps = vhr.BPM.BVPsignal(bvp_pred, self.fps)
        print("windowing BVP...")
        bvp_win, timesES = BVP_windowing(bvp_pred, self.wsize, self.fps, stride=1)
        print("converting BVP to BPM...")
        bpmES = BVP_to_BPM_2(bvp_win, self.fps) # BPM estimation
        print("bpmES: ", bpmES)
        print("len(bpmES): ", len(bpmES))
        if save:
            #np.save(f"Code/validate_models/bvpES_{method}.npy", bvp_pred)
            np.save(f"Code/validate_models/bpmES_{method}.npy", bpmES)
            return None
        else:
            return bpmES

if __name__ == "__main__":
    test = Test_Methods()
    test.test_deep(method = "MTTS_CAN")
    #test.test_deep(method = "HR_CNN")