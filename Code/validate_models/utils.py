from scipy.signal import butter, lfilter, welch
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import spdiags
import cv2
from skimage.util import img_as_float
from tqdm import tqdm


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psnr(img, img_g):
    criterionMSE = nn.MSELoss()  # .to(device)
    mse = criterionMSE(img, img_g)

    psnr = 10 * torch.log10(torch.tensor(1) / mse)  # 20 *
    return psnr

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def preprocess_raw_video(frames, fs=30, dim=36):
  """A slightly different version from the original: 
    takes frames as input instead of video path """

  totalFrames = frames.shape[0]
  Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
  i = 0
  t = []
  width = frames.shape[2]
  height = frames.shape[1]
  # Crop each frame size into dim x dim
  for img in frames:
    t.append(1/fs*i)       # current timestamp in milisecond
    img = img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]
    vidLxL = cv2.resize(img_as_float(img), (dim, dim), interpolation = cv2.INTER_AREA)
    vidLxL[vidLxL > 1] = 1
    vidLxL[vidLxL < (1/255)] = 1/255
    Xsub[i, :, :, :] = vidLxL
    i = i + 1
  #import matplotlib.pyplot as plt
  #plt.imshow(Xsub[0])
  #plt.show()

  # Normalized Frames in the motion branch
  normalized_len = len(t) - 1
  dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
  for j in range(normalized_len - 1):
    dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
  dXsub = dXsub / np.std(dXsub)
  
  # Normalize raw frames in the apperance branch
  Xsub = Xsub - np.mean(Xsub)
  Xsub = Xsub  / np.std(Xsub)
  Xsub = Xsub[:totalFrames-1, :, :, :]
  
  # Plot an example of data after preprocess
  dXsub = np.concatenate((dXsub, Xsub), axis=3);
  return dXsub

def extract_raw(videoFileName):
        """
        Extracts raw frames from video.

        Args:
            videoFileName (str): video file name or path.

        Returns: 
            ndarray: raw frames with shape [num_frames, height, width, rgb_channels].
        """

        frames = []
        for frame in tqdm(extract_frames_yield(videoFileName)):
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # convert to RGB

        return np.array(frames)

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()
  
def get_fps(videoFileName):
    """
    This method returns the fps of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps

def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)

def BVP_windowing(bvp, wsize, fps, stride=1):
  """ Performs BVP signal windowing

    Args:
      bvp (list/array): full BVP signal
      wsize     (float): size of the window (in seconds)
      fps       (float): frames per seconds
      stride    (float): stride (in seconds)

    Returns:
      bvp_win (list): windowed BVP signal
      timesES (list): times of (centers) windows 
  """
  
  bvp = np.array(bvp).squeeze()
  block_idx, timesES = sliding_straded_win_idx(bvp.shape[0], wsize, stride, fps)
  bvp_win  = []
  for e in block_idx:
      st_frame = int(e[0])
      end_frame = int(e[-1])
      wind_signal = np.copy(bvp[st_frame: end_frame+1])
      bvp_win.append(wind_signal[np.newaxis, :])

  return bvp_win, timesES

class BPM:
    """
    Provides BPMs estimate from BVP signals using CPU.

    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].
    """
    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz

    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, minHz=self.minHz, maxHz=self.maxHz, nfft=self.nFFT)
        # -- BPM estimate
        Pmax = np.argmax(Power, axis=1)  # power max
        return Pfreqs[Pmax.squeeze()]

def BVP_to_BPM_2(bvps, fps, minHz=0.65, maxHz=4.):
    """
    Computes BPMs from multiple BVPs (window) using PSDs maxima (CPU version)

    Args:
        bvps (list): list of length num_windows of BVP signal defined as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).

    Returns:
        A list of length num_windows of BPM signals defined as a float32 Numpy.ndarray with shape [num_estimators, ].
        If any BPM can't be found in a window, then the ndarray has num_estimators == 0.
        
    """
    bpms = []
    obj = None
    for bvp in bvps:
        if obj is None:
            obj = BPM(bvp, fps, minHz=minHz, maxHz=maxHz)
        else:
            obj.data = bvp
        bpm_es = obj.BVP_to_BPM()
        bpms.append(bpm_es)
    return bpms

def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power