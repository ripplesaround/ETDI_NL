CUDA_VISIBLE_DEVICES='2' python3 main.py \
    --run_mode big_loss \
    --model proposed \
    --num_gradual 10 \
    --lr 0.001 \
    --batch_size 128 \
    --num_class 10  \
    --noise_prob 0.4 \
    --noise_type asym \
    --dataset CIFAR10 \
    --epochs 30    