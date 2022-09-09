# -*- coding: utf-8 -*-


import librosa
import warnings
import os,pandas as pd,numpy as np
from joblib import dump, load
import math
# TORCH MODULES
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import sklearn

# FOR OLDER VERSIONS (OLD PICKLES, use this sklearn verion for svc old )
# !pip uninstall scikit-learn  -y && pip install scikit-learn==0.22.2

"""# Signal preprocessing functions"""

def signal_zero_padding(original_signal,desired_signal_size,verbose=0):
  """
  Perform signal_zero_padding adding the same (or almost, diff by 1) quantity of zeros to the left and to the right of the original_signal ; the final
  signal length will be equal to  desired_signal_size
  """
  original_signal_size = len(original_signal)
  if verbose>0:
    print("Original Signal length:",original_signal_size)
  x = (desired_signal_size-original_signal_size)/2
  signal_start,signal_end = math.floor(x),math.floor(x)+original_signal_size
  new_signal = np.zeros(desired_signal_size)
  new_signal[signal_start:signal_end] = original_signal
  return new_signal

def onset_to_stft(onset_padded, sampling_rate=22050,stride=4,window_size = 128):
  # from https://stackoverflow.com/questions/56286595/librosas-fft-and-scipys-fft-are-different
  # option 2
  #window_size = 128  # 2048-sample fourier windows
  #stride = 4      # 512 samples between windows
  wps = sampling_rate/float(stride) # ~86 windows/second
  X_libs = librosa.stft(onset_padded, n_fft=window_size, hop_length=stride)
  X_libs = np.abs(X_libs)[:,:int(2*wps)]
  # important
  # wsize = 64 ; so output expected is 64/2+1 =33 rows  (positive part of the fft)
  #  columns: depends on how much stride there is; stride = 64 ; signal of 128 --> 128/64 +1 = 3 ; stride = 16 --> 128/16 +1  =8  +1 = 9
  return X_libs

def onset_to_signal(onset_number, signal, sampling_rate, seconds_window, desired_signal_size_for_padding):
  """
  Parameters from the signal:
    signal: signal (not fft) np.1darray; 
    sampling_rate: sampling rate of the signal
    onset_number: the number of sample where the onset occurs in the signal
    Parameters to be tunned:
    seconds_window -> it goes some seconds forward and some seconds backward to get a chunk of the signal ; if 1 , from the onset goes 1 second forwar and 1 second backward and get that whole chunk as the newsignal
    desired_signal_size_for_padding -> If you desired the 1d signal to be any size you can perform  zero padding ; if = None then it does not perform any padding.
  returns ->
     a chunk of the signal (where the onset is +-seconds_window)
  """
  mseconds_samples = int(seconds_window*sampling_rate) # 36 samples = approximately 0.1 seconds
  # Redefine start,end and take signal
  onset_start,onset_end = max(0,onset_number-mseconds_samples),min(len(signal),onset_number+mseconds_samples)
  onset_signal = signal[onset_start:onset_end]
  if desired_signal_size_for_padding is not None:
    #zero padd signal
    onset_signal  = signal_zero_padding(onset_signal, desired_signal_size_for_padding)
  return onset_signal

def onset_to_features(onset_signal,sampling_rate, hop_size, n_fft, flatten=True):
  """
  onset_to_features for the models;
  onset_signal -> must be a 1d signal; we will perform stft in here
  sampling_rate -> sampling rate of the signal
  hop_size : stride or hop size of the stft  (see librosa stft)
  n_fft : size of the window of the moving fft (stft)
  flatten: if you want to flatten the results from the stft
  return -> preprocessed stftsignal and reshaped
  """
  # transform onset_signal chunk to stft
  onset_signal_stft = onset_to_stft(onset_signal, sampling_rate=sampling_rate,stride=hop_size,window_size = n_fft)
  if flatten:
    # flatten stft
    onset_signal_stft = onset_signal_stft.flatten()
    # reshape for being able to use .predict without any issue
    onset_signal_stft  = onset_signal_stft.reshape(1,-1)
  else:
    # This is if you dont pass a batch of files to predict ,just a single file instead
    onset_signal_stft = np.expand_dims(onset_signal_stft, axis=0)
  return onset_signal_stft

def predict_drumtypes_in_onset(onset_with_stft, sklearn_models_list, drop_other_preds = True):
  """
  sklearn_models_list: list of models, must have .predict method (ie sklearn or a torch nn class with the .predict method in it);
                       the miportant stuff here is that you can input 2 models and this will detect if any of the 2 drum types is present
  onset_with_stft: must be onset signal ready for the .predict method; sometimes need to .reshape(-1,1) 
  drop_other_preds, bool: if True will drop predictions labeled as OTHER (different from target); else will keep them
  return: drumtypes_present_in_onset (drumtypes preseint in the current onset; could be more than 1)
  """
  drumtypes_present_in_onset = list()
  for model in sklearn_models_list:
    pred = model.predict(onset_with_stft)
    #this is to fix a quick bug
    pred = np.array(pred).tolist()[0]
    if drop_other_preds:
      if pred != "OTHER":
        drumtypes_present_in_onset.append(pred)
    else:
      drumtypes_present_in_onset.append(pred)
  return drumtypes_present_in_onset

"""# Detector class"""

class DrumTypesDetector():
  def __init__(self, config_signal_params,sklearn_models_list, flatten_data, drop_other_preds = True, return_onset_times=True):
    """
    params:
      flatten_data: bool if date needs to be flatten or not
      sklearn_models_list: list of models for making the prediction, must have .predict method
      config_signal_params: dict with the config singlap arams in order to make the stft and sther stuff (depending on your data) 
                           ie config_signal_params = {"hop_size":256,"n_fft":1024,"desired_signal_size_for_padding":4096,"seconds_window":0.05}
      drop_other_preds: to drop labels as "OTHER".
      return_onset_times: to return onset_time instead of onset_sample; by default it's drue ; be aware that we are getting onset_time as a result of round(onset_number/sampling_rate,16) so it could have rounding issues
    """
    self.config_signal_params = config_signal_params
    #models list consistent with config_signal_params
    self.sklearn_models_list = sklearn_models_list
    # flatten data
    self.flatten_data =flatten_data
    # one way of format
    self.dict_formatted_onsets  = dict()
    #another way of formatting
    self.list_formatted_onsets = list()
    # return onsets in time format instead of integers (onset_sample)
    self.return_onset_times = return_onset_times
    #
    self.drop_other_preds = drop_other_preds

  def onset2drumtypes(self,signal,sampling_rate, onsets_all, hop_size, n_fft, desired_signal_size_for_padding, seconds_window):
    """
    For a list of onsets  <onsets_all >and a signal <signal>  this model will predict onset_drumtypes using <self.sklearn_models_list>
    We still need toenhance the docs
    """
    for onset_number in onsets_all:
      #onset_number = onsets_all[0]
      onset_number_time = round(onset_number/sampling_rate,16)
      # get the desired chunk of the signal
      onset_signal = onset_to_signal(onset_number, signal, sampling_rate, seconds_window, desired_signal_size_for_padding)
      onset_features = onset_to_features(onset_signal,sampling_rate, hop_size, n_fft, flatten=self.flatten_data)
      drumtypes_in_onset = predict_drumtypes_in_onset(onset_features,self.sklearn_models_list, drop_other_preds = self.drop_other_preds)
      
      if len(drumtypes_in_onset)>0:
        # format to seconds ; and add drumtype
        if self.return_onset_times:
          onset_returning = onset_number_time
        else:
          onset_returning = onset_number

        formatted_onset_drumtypes = [(onset_returning,drumtype) for drumtype in drumtypes_in_onset]
        # also with dict_format
        self.dict_formatted_onsets[f'{onset_returning}'] = drumtypes_in_onset
        self.list_formatted_onsets.extend(formatted_onset_drumtypes)
  def __call__(self,wav_path, presettled_onsets = None):
    """
    wav_path - str: wav path for the song will be loaded with librosa.load function
    presettled_onsets [default = None] - list: pressettled_onsets ->  corresponding onsets to this song; you can use your own onset detector for this song and put it in hre
                                                            . If settled to None, an onset detector from librosa will perform the onset detectino part. 
    return: Nothing but the output that you should call is self.dict_formatted_onsets or self.list_formatted_onsets
    """
    signal,sampling_rate = librosa.load(wav_path)

    # You can pass your onsets list to analize Within The song here (but must match with the drums in wav_path); ej if you pass an onset on second 39 a
    #and the song duration is 35 , will throw an error,, since  you are expecting to analyze an onset that is not in the song
    if presettled_onsets is None:
      presettled_onsets = librosa.onset.onset_detect(y=signal.astype('float32'), sr=sampling_rate, onset_envelope=None
                          , hop_length=128, backtrack=True, energy=None, units='samples')
  
    self.onset2drumtypes(signal=signal, sampling_rate=sampling_rate , onsets_all = presettled_onsets, **self.config_signal_params)

"""# Load models"""

def load_model(model_path):
  """
  Function for loading the model, must be .pth  from pytroch or .joblib from sklearn both must have a .predict method
  """
  file_name,file_suffix = os.path.splitext(model_path)
  if ".joblib" in file_suffix:
    clf = load(model_path) 
  elif ".pth" in file_suffix:
    clf = torch.load(model_path)
  else:
    raise ValueError("Please insert sklearn .joblib model or a torch .pth model")
  return clf

