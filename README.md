### TSleep - Automated Sleep Stage Scoring using Deep Learing

-------
+ Final project of "**CS598. Deep Learning for Healthcare**" course at [University of Illinois at Urbana-Champaign](https://illinois.edu/)
+ Team: `thucd2@illinois.edu` (1 member)

In this experiment, I construct two deep neural networks to score the sleep stage automatically. Both models are working on single-channel signals from Sleep Cassette study dataset in publicly available Sleep-EDF 2018 dataset. The first model comprises multiple convolutional neural networks (CNN) and bidirectional long short-term memory (Bi-LSTM) networks achieving overall accuracy of `75.4%`, an average macro F1-score of `67.8%` across all folds in 20-fold cross-validation and inter-rater reliability coefficient Cohen's kappa `κ = 0.66`. The second model is based on convolutional neural networks predicting with an average accuracy of `79%` and average macro F1-score of `75%` and `κ =0.70` in the same cross-validation procedure. The result shows that my model is comparable to state-of-the-art methods with hand-engineered features.

Read the full report [here](report/thucd2-sleep-project-final.pdf)

### Download

I evaluate the model with data from Sleep Cassette study of Sleep-EDF 2018 Dataset. Run the command below to download all data:
```
cd data
./download_sleep_edfx.sh
```

### Extract channels

Then run the following script to extract specified EEG channels and their corresponding sleep stages.
(this script was forked from [DeepSleepNet](https://github.com/akaraspt/deepsleepnet))
```
python prepare_physionet.py --data_dir data/sleep-cassette --output_dir data/sc_eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
python prepare_physionet.py --data_dir data/sleep-cassette --output_dir data/sc_eeg_pz_oz --select_ch 'EEG Pz-Oz'
```

### Training & evaluate model
```
python train.py  --model [model_name] --fold_ids [fold or list of folds] --data_dir [direction containing data] --output_dir [folder to output result, model]
```
Some examples
+ Train first fold of Modified Sleep EEG Net
```
python train.py --model mod_sleep_eeg --fold_ids 0 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ Train first two fold of Modified Sleep EEG Net
```
python train.py --model mod_sleep_eeg --fold_ids 0,1 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ Train last fold of Modified Deep Sleep Net
```
python train.py --model mod_deep_sleep --fold_ids 19 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
```

+ Train and evaluate all folds of Modified Sleep EEG Net
```
python train.py --model mod_sleep_eeg --fold_ids -1 --data_dir data/sc_eeg_fpz_cz --output_dir output/sc_eeg_fpz_cz
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
    + numpy 1.19.2
    + pandas `1.2.4`
    + scikit-learn `0.24.1`
    + mne `0.23.0`

Notice: I have faced some runtime issues with tensorflow `2.1.x`, and numpy `1.20`, so it's advised to create a conda enviroment from `environment.yml` to make sure you have the same libraries as mine.
You can use the script below:
```
conda env create -f environment.yml
```