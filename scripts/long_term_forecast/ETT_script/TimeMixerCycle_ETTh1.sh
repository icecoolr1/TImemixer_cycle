export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixerV2  # 修改模型名称

# ======== 核心参数优化 ========
seq_len=96
e_layers=3               # 增加编码层数
down_sampling_layers=2   # 减少下采样层数
down_sampling_window=4   # 增大下采样窗口
learning_rate=0.001      # 降低学习率
d_model=256              # 增大模型维度
d_ff=512                 # 增大前馈维度
train_epochs=50          # 增加训练轮次
patience=15              # 增大早停耐心值
batch_size=32            # 减小批次大小

# ======== 新增周期参数 ========
cycle_lens="24_48,168_84,720_360"  # 各尺度周期配置（尺度0:24+48h, 尺度1:168+84h, 尺度2:720+360h）
fusion_type="attention"           # 使用注意力融合
phase_dropout=0.2                 # 相位索引dropout
ortho_weight=0.05                 # 周期基正交约束强度

# ======== 通用参数 ========
common_args=(
  --task_name long_term_forecast
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh1.csv
  --model_id ETTh1_${seq_len}
  --model $model_name
  --data ETTh1
  --features M
  --seq_len $seq_len
  --label_len 0
  --enc_in 7
  --c_out 7
  --des 'Exp'
  --itr 3                        # 增加实验重复次数
  --d_model $d_model
  --d_ff $d_ff
  --learning_rate $learning_rate
  --train_epochs $train_epochs
  --patience $patience
  --batch_size $batch_size
  --down_sampling_layers $down_sampling_layers
  --down_sampling_window $down_sampling_window
  --down_sampling_method conv    # 改为可学习的卷积下采样
  --cycle_lens $cycle_lens       # 新增周期配置
  --fusion_type $fusion_type     # 融合策略
#  --phase_dropout $phase_dropout
#  --ortho_weight $ortho_weight
  --use_amp                      # 启用混合精度训练
)

# ======== 多预测长度测试 ========
pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
  python -u run.py "${common_args[@]}" \
    --pred_len $pred_len \
    --model_id ETTh1_${seq_len}_${pred_len}
done