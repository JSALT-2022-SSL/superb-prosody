# SUPERB-Prosody

## Introduction
Here is the official code for superb-prosody. The goal of this codebase is to examine the ability of Speech SSL (Self-Supervised Learning) in prosody field. There are four prosody-related tasks in this repo, which are **sarcasm detection**, **sentiment analysis**, **persuasiveness prediction** and **prosody reconstruction**.

There are five sections in this README.
- Environment setting
- Sarcasm Detection
- Sentiment Analysis
- Persuasiveness Prediction
- Prosody Reconstruction

## Environment setting

Please follow the below instruction:
```
# under /superb-prosody
cd s3prl
pip install -e .
pip install jsonlines audiomentations
...
```

## Sarcasm Detection
### Dataset preparation

1. Download data from [link](https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view) 
2. unzip the file under `/superb-prosody/s3prl/downstream/mustard/data/videos`

### Training
Use below script to train the sarcasm detection
```
# under /superb-prosody/s3prl

python run_downstream.py -m train -u fbank -d mustard
```

## Sentiment Analysis
### Dataset preparation
### Training


## Persuasiveness Prediction
### Dataset preparation
### Training


## Prosody Reconstruction
### Dataset preparation
### Training




