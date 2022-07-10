#!/bin/bash
export PATH="/home/twsewvd257/miniconda3/bin/:$PATH"
sudo apt update
sudo apt-get -y install libsndfile1
sudo apt-get -y install libsox-fmt-all libsox-dev sox
conda init
conda activate sarcasm
cd /home/twsewvd257/superb-prosody/s3prl
upstream=$1 #wav2vec2_base_960 #hubert_large_ll60k wav2vec2_large_ll60k hubert_base #vq_wav2vec
EXP_NAME="mustard-$upstream"
for lr in {1e-5,1e-4,1e-3}
do
  echo "Starting training $upstream with $lr"
  for i in {0..4}
  do
    python run_downstream.py \
      -m train -u $upstream -d mustard \
      -p result/$EXP_NAME-$lr/fold-$i \
      -o "config.downstream_expert.datarc.speaker_dependent=True,,config.downstream_expert.datarc.split_no=${i},,config.optimizer.lr=$lr"
  done
  python downstream/mustard/gather_score.py -p result/$EXP_NAME-$lr
done
conda deactivate