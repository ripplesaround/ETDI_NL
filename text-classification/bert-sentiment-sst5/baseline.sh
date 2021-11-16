CUDA_VISIBLE_DEVICES='2' python3 run.py\
    -c bert-base-uncased \
    -r \
    -e 30 \
    -nr 0.3 \
    -nt pairflip\
    -m coteaching