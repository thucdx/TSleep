## TSleep - Automated Sleep Stage Scoring using Deep Learning

-------
+ Final project of "**CS598. Deep Learning for Healthcare**" course at [University of Illinois at Urbana-Champaign](https://illinois.edu/)
+ Team: `thucd2@illinois.edu` (1 member)

In this experiment, I construct two deep neural networks to score the sleep stage automatically. Both models are working on single-channel signals from Sleep Cassette study dataset in publicly available Sleep-EDF 2018 dataset. The first model comprises multiple convolutional neural networks (CNN) and bidirectional long short-term memory (Bi-LSTM) networks achieving overall accuracy of `75.4%`, an average macro F1-score of `67.8%` across all folds in 20-fold cross-validation and inter-rater reliability coefficient Cohen's kappa `κ = 0.66`. The second model is based on convolutional neural networks predicting with an average accuracy of `79%` and average macro F1-score of `75%` and `κ =0.70` in the same cross-validation procedure. The result shows that my model is comparable to state-of-the-art methods with hand-engineered features.

Read the full report [here](report/thucd2-sleep-project-final.pdf)

### Download dataset

I evaluate the model with data from Sleep Cassette study of Sleep-EDF 2018 Dataset. Run the command below to download all data:
```
cd data
chmod +x download_sleep_edfx.sh
./download_sleep_edfx.sh
```

### Extract channels from dataset

Then run the following script to extract specified EEG channels and their corresponding sleep stages.
(this script was forked from [DeepSleepNet](https://github.com/akaraspt/deepsleepnet))
```
python prepare_physionet.py --data_dir data/sleep-cassette --output_dir data/sc_eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
python prepare_physionet.py --data_dir data/sleep-cassette --output_dir data/sc_eeg_pz_oz --select_ch 'EEG Pz-Oz'
```

### Training & evaluate model

```
usage: main.py [-h] [--run RUN] [--data_dir DATA_DIR]
                [--output_dir OUTPUT_DIR] [--fold_ids FOLD_IDS]
                [--model MODEL] [--total_fold TOTAL_FOLD]

optional arguments:
  -h, --help            show this help message and exit
  --run RUN             running mode: train / summarize.
                        + train: Training. Need to specfiy data_dir, output_dir, fold_ids, model and total_fold
                        + summarize: View result of the last run in which the result is stored in `output_dir`
  --data_dir DATA_DIR   directory contain extracted channels
  --output_dir OUTPUT_DIR
                        Directory to store model, progress, result
  --fold_ids FOLD_IDS   fold/folds to train, each valid fold from [0 to total_fold-1].
                        + Can be a single fold, for example: 0 - the first fold, or 19 - the last fold in 20-fold cross validation
                        + Can contain multiple folds, separate by comma `,`, for example: 0,1,2.
                        + -1 to train on all folds
  --model MODEL         model to train and evaluate performance
                        + mod_sleep_eeg: Modified SleepEEG - using CNNS
                        + mod_deep_sleep: Modified DeepSleep - using CNNs + Bi-LSTMs
  --total_fold TOTAL_FOLD
                        Number of fold
```

##### Some quick usage examples

+ Train & evaluate result on the first fold of Modified Sleep EEG Net
```
python main.py --model mod_sleep_eeg --fold_ids 0 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ Train & evaluate result on the first two folds of Modified SleepEEG Net
```
python main.py --model mod_sleep_eeg --fold_ids 0,1 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ Train & evaluate result on the last fold of Modified DeepSleep Net
```
python main.py --model mod_deep_sleep --fold_ids 19 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ Train and evaluate result on all folds of Modified SleepEEG Net
```
python main.py --model mod_sleep_eeg --fold_ids -1 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ View summary of performance of the last run of modified_sleep_eeg model
(Can only view summary if you finished training)
```
python main.py --run summarize --output_dir output/sc_eeg_fpz_cz/modified_sleep_eeg
```

### Environment

- My hardware: 
    + AMD (R) Ryzen 7 3700x 8 cores 16
threads
    + 32 GB RAM
    + GPU GeForce GTX 1660 6GB
    + 1TB SSD.

-  The software environment is as follows: 
    + python `3.7.10`
    + tensorflow/tensorflow-gpu `2.4.1`
    + numpy `1.19.2`
    + pandas `1.2.4`
    + scikit-learn `0.24.1`
    + mne `0.23.0`

Notice: I have faced some runtime issues with tensorflow `2.1.x`, and numpy `1.20`, so it's advised to create a conda enviroment from `environment.yml` to make sure you have the same libraries as mine.
You can use the script below:
```
conda env create -f environment.yml
```