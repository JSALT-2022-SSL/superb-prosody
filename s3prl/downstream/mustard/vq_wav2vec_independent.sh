for lr in {1e-3,1e-4,1e-5}
do
    echo "Doing vq_wav2vec with $lr..."
    python run_downstream.py \
    -m train -u vq_wav2vec -d mustard \
    -p "result/speaker-independent_0621/vq_wav2vec-batch16acc2-no-aug-$lr-3000" \
    -o "config.optimizer.lr=$lr,,config.downstream_expert.datarc.train_batch_size=16,,config.runner.gradient_accumulate_steps=2"
done