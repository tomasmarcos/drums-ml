# !pip install madmom
import os
import pandas as pd
import numpy as np
import warnings

def load_labels(annotation_path_txt, set2df = True):
  """
  load labels given an annotation path
  Params:
    annotation_path, str: fullpath to annotation
  Return:
    pandas dataframe; with onset_time and drum_type columns
  """
  with open(annotation_path_txt,"r") as f:
    labels = [x.replace("\n","").strip().split(" \t ") for x in f.readlines()]
  if set2df:
    labels = pd.DataFrame(labels, columns = ["onset_time","drum_type"])
  return labels

def load_labels_just_onsets(annotation_path):
  """
  Load instead of labels + onset_time, just onset time ; given the annotation path
  """
  df_labels = load_labels(annotation_path, set2df = True)
  onset_time = df_labels["onset_time"].astype(float).tolist()
  return onset_time

def search_correspondingpath_given_annotation(annotation_path_txt, audio_directory):
  """
  This function will receive an annotation_txt path and look for its corresponding .wav inside the audio_directory ; audio_directory must have train and test folders 
  audio_directory is often -> AUDIO_DIR = "/content/drive/My Drive/Maestria DM y KDD/Especializacion tesis/MDBDrums/MDB Drums/audio/drum_only"
  from ESP_Pipe003_01_GenerateBarsFor3Different models.ipynb

  Params:
    annotation_path_txt, str: fullpath to annotation 
    audio_directory, str: where you keep your drum_only audios of MDB Drums dataset.
              ie: AUDIO_DIR = "./Especializacion tesis/MDBDrums/MDB Drums/audio/drum_only"
  Return:
    path of  the corresponding wav coming from annotation_path_txt

  """
  filepath_wav = os.path.basename(annotation_path_txt).replace(".txt",".wav")
  filepath_wav = filepath_wav.replace("_class","_Drum")
  if "test" in annotation_path_txt: #means need to go to perform the search where the ttest set is
    filepath_wav = os.path.join(audio_directory,"test",filepath_wav)
  elif "train" in annotation_path_txt:#means need to go to perform the search where the train  set is
    filepath_wav = os.path.join(audio_directory,"train",filepath_wav)
  else:
    raise ValueError("must be either in test or train , review the file")
  assert os.path.exists(filepath_wav),"The file doesn't exist , review your annotation process"
  return filepath_wav


class ComputeMetrics():
  def __init__(self, true_labels: pd.DataFrame, predicted_labels: pd.DataFrame, samples_criterion: int, filter_drumtype: str = None):
    """
    true_labels: pd.DataFrame with columns: annotation_path  ; onset_sample ; (and drum_type in case filter_drumtype is not None)
    predicted_labels: pd.DataFrame with columns:annotation_path ; onset_sample ; (and drum_type in case filter_drumtype is not None)
    annotation_path must match in true_labels and predicted_labels
    samples_criterion :
                      A detected onset is considered correct if the absolute time difference with the associated ground truth onset does not exceed 30 ms. 
                      +- some samples (equal to seconds window but this consider the sampling rate) ; equal to seconds_window/sampling_rate ; by default is 0.03/22050 which means
                      30 ms according to Jacques & Roebelm , but we can change this
    """
    #args
    self.true_labels = true_labels
    self.predicted_labels = predicted_labels
    self.samples_criterion = samples_criterion
    self.filter_drumtype = filter_drumtype
    #stored params
    # 
    self.recall, self.precision, self.f1_score = None, None, None

  def calculate_matches(self, onset_df_to, onsets_df_from, samples_criterion, proportion = False):
    """
    This is pretty straightforward: 
      for each onset in the dataset called onsets_df_from, look where the onset occurs in the signal and name this onset_detected; build a range onset_detected+-samples_criterion
      then see if there is any onset in onset_df_to within the range. 
      So the idea would be look from the preedted_onsets dataset and go if there is the corresponding onset on the real_onsets (labeled) and viceversa;
    samples_criterion :
                      A detected onset is considered correct if the absolute time difference with the associated ground truth onset does not exceed 30 ms. 
                      +- some samples (equal to seconds window but this consider the sampling rate) ; equal to seconds_window/sampling_rate ; by default is 0.03/22050 which means
                      30 ms according to Jacques & Roebelm , but we can change this
    proportion: if true, calculates the proportion from one dataset to another (that way you would calcualate the recall and the precision depending your onsets_dffrom and onset_df_to)
    """
    onset_df_to = onset_df_to.copy()
    onset_df_to.index = onset_df_to["onset_sample"]
    matches = 0
    n_onsets = len(onsets_df_from)
    for onset_idx in range(n_onsets):
      onset_detected = onsets_df_from.loc[onset_idx,"onset_sample"]
      # do the onset_samples +- time window
      # this is not needed since it doesnt matter if it exceeds the signal as long as there is an onset within the window
      #onset_detected_start, onset_detected_end = max(0,onset_detected-samples_criterion), min(len(signal), onset_detected+samples_criterion)
      onset_detected_start, onset_detected_end = onset_detected-samples_criterion, onset_detected+samples_criterion
      # therefore we redefine it like this to get rid of the signal parameter
      # once you have onset start,end go to the other dataset to see if there is an onset within that interval; if yes, then you have a TRUE POSITIVE
      # what if there are 2 onsets within that range? -- we wrote notes about
      #this.
      n_detected_onsets_within_range = len(onset_df_to.loc[onset_detected_start:onset_detected_end])
      if n_detected_onsets_within_range == 1:
        matches += 1
      elif n_detected_onsets_within_range>1:
        matches +=1
        warnings.warn("[WARNING] n_detected_onsets_within_range > 1!")

    if proportion:
      matches /= n_onsets
    return matches, n_onsets

  def calculate_precision_and_recall(self):
    unique_annotation_paths = self.true_labels["annotation_path"].unique().tolist()
    # print(unique_annotation_paths)
    total_matches_precision, total_onsets_precision = list(), list()
    total_matches_recall, total_onsets_recall = list(), list()
    for path in unique_annotation_paths:
      # get true and predicted labels from attributes
      true_labels = self.true_labels
      predicted_labels = self.predicted_labels
      #filter labels using annotation path
      true_labels = true_labels[true_labels["annotation_path"] == path].reset_index(drop=True)
      predicted_labels = predicted_labels[predicted_labels["annotation_path"] == path].reset_index(drop=True)
      if self.filter_drumtype is not None:
          # print(f"[INFO] Filtering {self.filter_drumtype} since self.filter_drmtypes is not None")
          true_labels = true_labels.loc[true_labels["drum_type"] == self.filter_drumtype].reset_index(drop=True)
          predicted_labels = predicted_labels.loc[predicted_labels["drum_type"] == self.filter_drumtype].reset_index(drop=True)
      # else , perform the filtering from predidicted labels and true_labels
      matches_precision, n_onsets_precision  = self.calculate_matches(onset_df_to = true_labels
                                                  , onsets_df_from = predicted_labels
                                                  , samples_criterion = self.samples_criterion
                                                  , proportion = False)
    
      matches_recall, n_onsets_recall  = self.calculate_matches(onset_df_to = predicted_labels
                                                  , onsets_df_from = true_labels
                                                  , samples_criterion = self.samples_criterion
                                                  , proportion = False)
  
      # now append each total for precision and recall
      total_matches_precision.append(matches_precision)
      total_onsets_precision.append(n_onsets_precision)
      total_matches_recall.append(matches_recall)
      total_onsets_recall.append(n_onsets_recall)

    self.recall_denominator = np.sum(total_onsets_recall)
    self.precision_denominator = np.sum(total_onsets_precision)
    self.precision = np.sum(total_matches_precision)/self.precision_denominator
    self.recall = np.sum(total_matches_recall)/self.recall_denominator
    return self.precision,self.recall

  def calculate_f1_score(self):
    if self.precision is not None and self.recall is not None:
      f1 = 2 * (self.precision * self.recall )/(self.precision + self.recall)
      self.f1_score = f1
    return self.f1_score
    
  def __call__(self):
    precision, recall = self.calculate_precision_and_recall()
    f1_score = self.calculate_f1_score()
    metrics_dict = {"recall": recall, "precision": precision, "f1_score": f1_score}
    return metrics_dict