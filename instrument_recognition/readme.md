# Scripts / Notebooks for training the models 

1)First of all you should execute preprocessing_notebook.ipynb and then save the preprocessed results.
2) After that, you'll be able to use one of the models (svc or cnn): (i) load the preprocessed signals
from (1); then train the model for the instrument. Here we input sample example for HH, for training
another instrument change target_label parameter to any of the following ones: KD, SD, OT, TT, CY, HH.


# Notes

1) When you execute the CNN, be aware that the perma dropout layer in it is a source of randomness
not controlled yet. However, you'll be able to achive similar results.

2)  This is a preliminary version.
