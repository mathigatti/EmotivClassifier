# EmotivClassifier

This are a few scripts to generate correlation matrices and then use them to train classifiers in order to predict different individual characteristics based on the brain connectivity. These are part of my computer science thesis project.

## Requirements

- python 3
- python libraries
  - sklearn
  - scipy
  - pandas
  - mne

## Usage
It is used running the main script `midi2voice.py`, it has four parameters, the lyrics_file, midi_file, singer sex (optional) and tempo (optional).

### Generating Correlation Matrices

In order to generate correlation matrices you need raw data from an EEG, in this case the scripts are prepared for EEGLab data format ( .set files ) with 14 time series (electrodes).

The script is run like this:

```
python correlationMatrixGenerator.py metric path_to_raw_eeg_data destination_path
```

### Generating Correlation Matrices

On classifier.py you can find a script to train a random forest classifier using correlation matrices as input in order to predict some variable from a CSV file.

```
python classifier.py path_to_correlation_matrices target_feature_path column_name threshold 
```

Here is an example training it to predict people with more than 5 hours of sleep using correlation matrices (Computed using spearman correlation on the theta band in this case).

```
python classifier.py matrices/spearman/theta/ users_data/individual_characteristics.csv "Hours of Sleep" 5 
```
