CUDA_VISIBLE_DEVICES='0' python3 main.py \
        --dataset cifar100 \
        --noise_type symmetric  \
        --noise_rate 0.5    \
        --lr 0.001       \
        --n_epoch 60
