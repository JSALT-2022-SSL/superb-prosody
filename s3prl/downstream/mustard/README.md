# Sarcasm Detection

## Download Data
Download video clips from this [link](https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view) to `data` folder and unzip. 

## Extract Audio
```
bash extract_audio_files.sh
```

## Training
```
python run_downstream.py -m train -u fbank -d mustard -n ExpName
```