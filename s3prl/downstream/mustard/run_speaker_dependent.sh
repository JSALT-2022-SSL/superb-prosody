EXP_NAME=mustard-hubert

cd ../..
for i in {2..4}
do
  echo "Fold ${i}"
  python run_downstream.py \
    -m train -u hubert -d mustard \
    -n $EXP_NAME-fold-$i \
    -o "config.downstream_expert.datarc.speaker_dependent=True,,config.downstream_expert.datarc.split_no=${i}"
done
python downstream/mustard/gather_score.py -n $EXP_NAME