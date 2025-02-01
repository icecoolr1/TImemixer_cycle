import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class RecurrentCycle(nn.Module):
    """增强版周期模块"""

    def __init__(self, cycle_len, d_model):
        super().__init__()
        self.cycle_len = cycle_len
        self.d_model = d_model
        self.base = nn.Parameter(torch.randn(cycle_len, d_model))
        self.phase_shift = nn.Linear(1, 1, bias=False)  # 可学习相位偏移

    def forward(self, phase, length):
        phase = self.phase_shift(phase.float().unsqueeze(-1)).squeeze()  # 学习相位调整
        gather_idx = (phase.long().view(-1, 1) + torch.arange(length, device=phase.device)) % self.cycle_len
        return self.base[gather_idx]


class MultiScaleCycleBlock(nn.Module):
    """多周期融合块"""

    def __init__(self, cycle_lens, d_model, fusion_type='gated'):
        super().__init__()
        self.cycles = nn.ModuleList([
            RecurrentCycle(clen, d_model) for clen in cycle_lens
        ])
        self.fusion_type = fusion_type

        if fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(d_model * len(cycle_lens), len(cycle_lens)),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'attention':
            self.attn = nn.MultiheadAttention(d_model, num_heads=4)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, phase, length):
        outputs = [cycle(phase, length) for cycle in self.cycles]

        if self.fusion_type == 'sum':
            fused = sum(outputs)
        elif self.fusion_type == 'gated':
            gates = self.gate(torch.cat(outputs, dim=-1))
            fused = sum(gates[:, :, i:i + 1] * out for i, out in enumerate(outputs))
        elif self.fusion_type == 'attention':
            fused, _ = self.attn(
                outputs[0].permute(1, 0, 2),
                torch.stack(outputs, dim=1).permute(1, 0, 2),
                torch.stack(outputs, dim=1).permute(1, 0, 2)
            )
            fused = fused.permute(1, 0, 2)

        return self.norm(fused)


class Model(nn.Module):
    """无季节分解的纯周期趋势模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # 输入嵌入
        self.enc_embedding = DataEmbedding_wo_pos(
            1 if configs.channel_independence else configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # 多尺度下采样
        self.downsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    configs.enc_in if i == 0 else configs.enc_in,
                    configs.enc_in,
                    kernel_size=3,
                    stride=configs.down_sampling_window,
                    padding=1,
                    padding_mode='circular'
                ),
                nn.GELU()
            ) for i in range(configs.down_sampling_layers)
        ])

        # 趋势处理核心
        self.trend_blocks = nn.ModuleList([
            MultiScaleCycleBlock(
                cycle_lens=configs.cycle_lens[i],
                d_model=configs.d_model,
                fusion_type=configs.fusion_type
            ) for i in range(configs.down_sampling_layers + 1)
        ])

        # 跨尺度混合
        self.cross_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model)
            ) for _ in range(configs.down_sampling_layers)
        ])

        # 预测头
        self.predictors = nn.ModuleList([
            nn.Linear(
                configs.seq_len // (configs.down_sampling_window ** i),
                configs.pred_len
            ) for i in range(configs.down_sampling_layers + 1)
        ])

        # 归一化
        self.norms = nn.ModuleList([
            Normalize(configs.enc_in, affine=True)
            for _ in range(configs.down_sampling_layers + 1)
        ])

    def _downsample(self, x):
        """生成多尺度输入"""
        scales = [x.permute(0, 2, 1)]
        for down in self.downsamplers:
            scales.append(down(scales[-1]))
        return [s.permute(0, 2, 1) for s in scales]

    def forward(self, x_enc, x_mark_enc,dec_inp, batch_y_mark, phase_indices):
        # 多尺度处理
        x_scales = self._downsample(x_enc)

        # 趋势处理
        trend_outputs = []
        for i, x in enumerate(x_scales):
            x_norm = self.norms[i](x, 'norm')
            emb = self.enc_embedding(x_norm, x_mark_enc)
            trend = self.trend_blocks[i](phase_indices[:, i], emb.size(1))
            trend_outputs.append(trend)

        # 跨尺度混合
        mixed = [trend_outputs[0]]
        for i in range(1, len(trend_outputs)):
            fused = self.cross_mixers[i - 1](
                torch.cat([mixed[-1], trend_outputs[i]], dim=-1)
            )
            mixed.append(fused)

        # 多尺度预测融合
        preds = []
        for i, trend in enumerate(mixed):
            pred = self.predictors[i](trend.permute(0, 2, 1)).permute(0, 2, 1)
            pred = self.norms[i](pred, 'denorm')
            preds.append(pred)

        return sum(preds) / len(preds)

    def _get_name(self):
        return 'TimeMixerV2'