CUDA_VISIBLE_DEVICES='2' python3 main.py \
    --run_mode train \
    --model normal \
    --num_gradual 15 \
    --lr 0.001 \
    --batch_size 128 \
    --num_class 10  \
    --noise_prob 0.5 \
    --noise_type sym \
    --dataset CIFAR10