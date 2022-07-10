#!/bin/bash
export PATH="/home/twsewvd257/miniconda3/bin/:$PATH"
sudo apt update
sudo apt-get -y install libsndfile1
sudo apt-get -y install libsox-fmt-all libsox-dev sox
conda init
conda activate sarcasm
cd /home/twsewvd257/superb-prosody/s3prl
upstream=$1
EXP_NAME="mustard-$upstream"
lr="1e-3"
echo "Starting training dependent setup $upstream layers 0,1,-1 with $lr"
for i in {0..4}
do
  python run_downstream.py \
      -m train -u $upstream -d mustard \
      -p result/$EXP_NAME-$lr-01x/fold-$i \
      -o "config.downstream_expert.datarc.speaker_dependent=True,,config.downstream_expert.datarc.split_no=${i},,config.optimizer.lr=$lr" \
      -s hidden_state_01x
done
python downstream/mustard/gather_score.py -p result/$EXP_NAME-$lr-01x