CUDA_VISIBLE_DEVICES='3' python3 main.py \
        --dataset cifar10 \
        --noise_type pairflip  \
        --noise_rate 0.4    \
        --n_epoch 40
