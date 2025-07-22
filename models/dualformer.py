import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.RevIN import RevIN
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Transformer_EncDec import EncoderLayer
from torch.fft import rfft, irfft


class HierarchicalFreqSampler(nn.Module):
    def __init__(self, seq_len, n_layers, alpha=1.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.alpha = alpha

    def sample(self, X_fft, layer_idx):
        M = X_fft.shape[1]
        F = int(self.alpha * M)
        if self.alpha > 1 / self.n_layers:
            pl = int(M * (1 - self.alpha) * (1 - (layer_idx) / (self.n_layers - 1)))
            ql = pl + F
        else:
            pl = int(M * (1 - (layer_idx + 1) / self.n_layers))
            ql = pl + int(M / self.n_layers)
        return X_fft[:, pl:ql]

class PeriodicWeight(nn.Module):
    def __init__(self, harmony_num=3):
        super().__init__()
        self.harmony_num = harmony_num

    def forward(self, x_enc):
        # Step 1: FFT and normalize
        freq = torch.fft.rfft(x_enc - x_enc.mean(dim=1, keepdim=True), dim=1)
        freq = torch.abs(freq)  # shape: [B, F, D]

        # Step 2: Suppress DC and clip tail
        _freq = freq.clone()
        _freq[:, :3, :] = 0  # remove low freq
        _freq[:, freq.shape[1] // self.harmony_num:, :] = 0  # remove high freq

        # Step 3: Max freq index per feature
        max_amp, indices = torch.max(_freq, dim=1, keepdim=True)  # [B, 1, D]
        amp_sum = torch.zeros_like(max_amp).to(x_enc.device)

        for i in range(self.harmony_num):
            har = (i + 1) * indices  # harmonic positions: 1f, 2f, 3f
            har = torch.clamp(har, max=freq.shape[1] - 1)
            har_value = torch.gather(freq, 1, har) ** 2
            amp_sum = amp_sum + har_value

        # Step 4: Normalize
        total_sum = torch.sum(freq ** 2, dim=1, keepdim=True) + 1e-5  # [B, 1, D]
        weights = amp_sum / total_sum  # [B, 1, D]

        return weights.transpose(1, 2)  # [B, D, 1] → so broadcasting with [B, L, D] works
    
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.n_layers = configs.e_layers
        self.revin = RevIN(configs.enc_in, affine=True)
        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.freq_sampler = HierarchicalFreqSampler(self.seq_len, self.n_layers, alpha=1.0)
        self.period_weight = PeriodicWeight()

        self.time_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(self.n_layers)
        ])

        self.freq_layers = nn.ModuleList([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(self.n_layers)
        ])

        self.time_norm = nn.LayerNorm(configs.d_model)
        self.freq_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)

    def ifft_with_pad(self, x_fft, target_len):
        # Recover original shape
        x_rec = irfft(x_fft, n=target_len, dim=1)
        if x_rec.shape[1] < target_len:
            pad_len = target_len - x_rec.shape[1]
            x_rec = F.pad(x_rec, (0, 0, 0, pad_len), mode='constant', value=0)
        return x_rec

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        x_enc = self.revin(x_enc, 'norm')
        emb = self.embedding(x_enc, x_mark_enc)
        x_fft = rfft(emb, dim=1)

        time_out, freq_out = emb, emb

        for i in range(self.n_layers):
            sampled_fft_time = self.freq_sampler.sample(x_fft, layer_idx=i)
            sampled_fft_freq = self.freq_sampler.sample(x_fft, layer_idx=i)

            time_input = self.ifft_with_pad(sampled_fft_time, self.seq_len)
            freq_input = self.ifft_with_pad(sampled_fft_freq, self.seq_len)

            time_out, _ = self.time_layers[i](time_input, attn_mask=None)
            freq_out, _ = self.freq_layers[i](freq_input, attn_mask=None)

        time_out = self.time_norm(time_out)
        freq_out = self.freq_norm(freq_out)

        weights = self.period_weight(emb)
        weights = weights.permute(0,2,1)

        fused = freq_out * weights + time_out * (1 - weights)
        fused = fused[:,-1,:]
        output = self.projection(fused)
        B, _ = fused.shape
        output = output.view(B, self.pred_len, self.c_out)
        output = self.revin(output, 'denorm')
        return output
