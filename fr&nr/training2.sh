CUDA_VISIBLE_DEVICES='3' python3 main.py \
        --dataset cifar10 \
        --noise_type symmetric  \
        --noise_rate 0.3    \
        --forget_rate 0.3    \
        --n_epoch 60
