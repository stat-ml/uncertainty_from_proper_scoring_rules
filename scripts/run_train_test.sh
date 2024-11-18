uv run python experiments/train_test_loop/main.py \
    --dataset cifar100 \
    --model resnet18_dropout \
    -c 0 \
    -v --log_to_file