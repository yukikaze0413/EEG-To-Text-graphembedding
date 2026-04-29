# 请将下方的 checkpoint_path 和 config_path 替换为您实际训练跑出来的模型路径
CUDA_VISIBLE_DEVICES=0 python eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1_task2_task3_finetune_BrainTranslator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path config/decoding/task1_task2_task3_finetune_BrainTranslator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0
