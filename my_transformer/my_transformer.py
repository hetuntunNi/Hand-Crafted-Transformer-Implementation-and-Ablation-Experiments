import torch
from torch import nn
import torch.nn.functional as F
import math


# ========== Token Embedding ==========
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


# ========== Positional Encoding（可关闭） ==========
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        div_term = torch.exp(_2i * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


# ========== Transformer Embedding（增加use_positional_encoding） ==========
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device, use_positional_encoding=True):
        super(TransformerEmbedding, self).__init__()
        self.use_positional_encoding = use_positional_encoding
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        if self.use_positional_encoding:
            pos_emb = self.pos_emb(x)
            return self.drop_out(tok_emb + pos_emb)
        else:
            return self.drop_out(tok_emb)


# ========== Multi-Head Attention ==========
class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MutiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        out = self.w_combine(score)
        return out


# ========== LayerNorm ==========
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


# ========== FeedForward ==========
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ========== Encoder Layer（增加use_residual） ==========
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1, use_residual=True):
        super(EncoderLayer, self).__init__()
        self.attention = MutiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        if self.use_residual:
            x = self.norm1(x + _x)
        else:
            x = self.norm1(x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        if self.use_residual:
            x = self.norm2(x + _x)
        else:
            x = self.norm2(x)
        return x


# ========== Encoder ==========
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head,
                 n_layer, drop_prob, device, use_positional_encoding=True, use_residual=True):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(
            enc_voc_size, d_model, max_len, drop_prob, device, use_positional_encoding
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob, use_residual)
             for _ in range(n_layer)]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


# ========== Decoder Layer（增加use_residual） ==========
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, use_residual=True):
        super(DecoderLayer, self).__init__()
        self.attention1 = MutiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.cross_attention = MutiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
        self.use_residual = use_residual

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        if self.use_residual:
            x = self.norm1(x + _x)
        else:
            x = self.norm1(x)

        _x = x
        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        if self.use_residual:
            x = self.norm2(x + _x)
        else:
            x = self.norm2(x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        if self.use_residual:
            x = self.norm3(x + _x)
        else:
            x = self.norm3(x)

        return x


# ========== Decoder ==========
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head,
                 n_layer, drop_prob, device, use_positional_encoding=True, use_residual=True):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(
            dec_voc_size, d_model, max_len, drop_prob, device, use_positional_encoding
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob, use_residual)
             for _ in range(n_layer)]
        )
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        dec = self.fc(dec)
        return dec


# ========== Transformer 主体 ==========
class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 drop_prob,
                 device,
                 use_positional_encoding=True,
                 use_residual=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden,
                               n_heads, n_layers, drop_prob, device,
                               use_positional_encoding, use_residual)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden,
                               n_heads, n_layers, drop_prob, device,
                               use_positional_encoding, use_residual)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # --- mask functions ---
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        batch_size, len_q = q.size()
        batch_size, len_k = k.size()
        q_mask = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3).expand(batch_size, 1, len_q, len_k)
        k_mask = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, len_q, len_k)
        return q_mask & k_mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k, device=self.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask.bool()

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)
        casual_mask = self.make_casual_mask(trg, trg)
        trg_mask = trg_pad_mask & casual_mask

        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out
