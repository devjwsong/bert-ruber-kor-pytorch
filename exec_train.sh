python src/main.py \
    --seed=555 \
    --model_path=MODEL_PATH \
    --default_root_dir="." \
    --data_dir="data" \
    --max_len=256 \
    --num_epochs=5 \
    --train_batch_size=32 \
    --eval_batch_size=16 \
    --num_workers=4 \
    --warmup_ratio=0.1 \
    --max_grad_norm=1.0 \
    --learning_rate=5e-5 \
    --gpus="3" \
    --pooling="cls" \
    --w1_size=768 \
    --w2_size=256 \
    --w3_size=64 \
    --num_hists=0
