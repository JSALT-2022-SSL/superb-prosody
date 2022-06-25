upstream=$1
EXP_NAME="mustard-$upstream"
for lr in {1e-5,1e-4,1e-3}
do
  for i in {0..4}
  do
    python run_downstream.py \
      -m train -u $upstream -d mustard \
      -p result/$EXP_NAME-$lr/fold-$i \
      -o "config.downstream_expert.datarc.speaker_dependent=True,,config.downstream_expert.datarc.split_no=${i},,config.optimizer.lr=$lr"
  done
  python downstream/mustard/gather_score.py -p result/$EXP_NAME-$lr
done