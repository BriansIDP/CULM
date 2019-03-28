export CUDA_VISIBLE_DEVICES=3 #${X_SGE_CUDA_DEVICE}
# export PATH="/home/miproj/urop.2018/gs534/Software/anaconda3/bin:$PATH"

python train_with_dataloader.py \
    --data data/AMI/ \
    --cuda \
    --emsize 256 \
    --nhid 768 \
    --dropout 0.5 \
    --rnndrop 0.25 \
    --epochs 30 \
    --lr 10 \
    --clip 0.25 \
    --nlayers 1 \
    --batch_size 64 \
    --bptt 12 \
    --wdecay 5e-6 \
    --model LSTM \
    --reset 1 \
    --logfile LOGs/rnn1L.log \
    --save models/model.12.1L.pt

