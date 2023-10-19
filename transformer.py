import torch
import torch.nn as nn
from config import Config
import numpy as np

config = Config()


class Transformers(nn.Module):
    def __init__(self):
        super(Transformers, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder().to(config.device)
        self.propetice = nn.Linear(config.model, config.word_count, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        encoder = self.encoder.to(config.device)
        decoder = self.decoder.to(config.device)
        enc_outputs = encoder(enc_inputs)
        dec_outputs = decoder(enc_inputs, enc_outputs, dec_inputs)
        dec_logits = self.propetice(dec_outputs)
        return dec_logits.view(-1, config.word_count)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embded = nn.Embedding(config.word_count, config.model)
        self.pos_embed = nn.Embedding.from_pretrained(positional_embeding(config.word_count, config.model))
        self.layers = nn.ModuleList([Encoderlayer() for _ in range(config.layers)])

    def forward(self, inputs):
        enc_outputs = self.embded(inputs) + self.pos_embed(inputs)
        pad_mask = self_attn(inputs, inputs)
        for layer in self.layers:
            layer = layer.to(config.device)
            enc_outputs = layer(enc_outputs, pad_mask)
        return enc_outputs


class Encoderlayer(nn.Module):
    def __init__(self):
        super(Encoderlayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_dff = Feed_Forward()

    def forward(self, enc_outputs, pad_mask):
        enc_self_attn = self.enc_self_attn.to(config.device)
        enc_output = enc_self_attn(enc_outputs, enc_outputs, enc_outputs, pad_mask)
        enc_output = self.pos_dff(enc_output)
        return enc_output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_embed = nn.Embedding(config.word_count, config.model)
        self.pos_embed = nn.Embedding.from_pretrained(positional_embeding(config.word_count, config.model))
        self.layers = nn.ModuleList([Decoderlayer() for _ in range(config.layers)])

    def forward(self, enc_inputs, enc_ouputs, dec_inputs):
        dec_outputs = self.dec_embed(dec_inputs)
        dec_self_attn = self_attn(dec_inputs, dec_inputs)
        dec_sub_attn = self_dec_mask(dec_inputs)
        dec_enc_attn = self_attn(dec_inputs, enc_inputs)
        dec_self_attn, dec_sub_attn = dec_self_attn.to(config.device), dec_sub_attn.to(config.device)
        dec_self_attn_mask = torch.gt((dec_self_attn + dec_sub_attn), 0)
        for layer in self.layers:
            layer = layer.to(config.device)
            dec_outputs = layer(enc_ouputs, dec_outputs, dec_enc_attn, dec_self_attn_mask)
        return dec_outputs


class Decoderlayer(nn.Module):
    def __init__(self):
        super(Decoderlayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_dff = Feed_Forward()

    def forward(self, enc_outputs, dec_outputs, dec_enc_attn1, dec_self_attn_mask):
        dec_self_attn = self.dec_self_attn.to(config.device)
        dec_enc_attn = self.dec_enc_attn.to(config.device)
        dec_output = dec_self_attn(dec_outputs, dec_outputs, dec_outputs, dec_self_attn_mask)
        dec_output = dec_enc_attn(dec_output, enc_outputs, enc_outputs, dec_enc_attn1)
        dec_output = self.pos_dff(dec_output)

        return dec_output


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(config.model, config.heads * config.d_q)
        self.W_K = nn.Linear(config.model, config.heads * config.d_k)
        self.W_V = nn.Linear(config.model, config.heads * config.d_v)
        self.socre = Scale()
        self.linear = nn.Linear(config.heads * config.d_q, config.model)
        self.layernorm = nn.LayerNorm(config.model)

    def forward(self, Q, K, V, pad_mask):
        q = self.W_Q(Q).view(Q.size()[0], Q.size()[1], config.heads, config.d_q).transpose(1, 2)
        k = self.W_K(K).view(K.size()[0], K.size()[1], config.heads, config.d_k).transpose(1, 2)
        v = self.W_V(V).view(V.size()[0], V.size()[1], config.heads, config.d_v).transpose(1, 2)
        pad_mask = pad_mask.unsqueeze(1).repeat(1, config.heads, 1, 1)
        socre = self.socre.to(config.device)
        context = socre(q, k, v, pad_mask)
        context = context.transpose(1, 2).contiguous().view(Q.size()[0], -1, config.heads * config.d_q)
        context = self.linear(context)
        context = self.layernorm(context + Q)
        return context


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, pad_mask):
        attention = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(config.d_k)
        attention.masked_fill(pad_mask, -1e9)
        attention = self.softmax(attention)
        context = torch.matmul(attention, v)

        return context


class Feed_Forward(nn.Module):
    def __init__(self):
        super(Feed_Forward, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=config.model, out_channels=config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=config.d_ff, out_channels=config.model, kernel_size=1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(config.model)

    def forward(self, enc_outputs):
        output = self.conv1(enc_outputs.transpose(1, 2))
        output = self.relu(output)
        output = self.conv2(output).transpose(1, 2)
        output = self.layernorm(output + enc_outputs)
        return output


def self_attn(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_mask_attn = seq_k.eq(0).unsqueeze(1)
    return pad_mask_attn.expand(batch_size, len_q, len_k)


def self_dec_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


def positional_embeding(n_position, d_model):
    def pos_embed(pos, d_model):
        return [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]

    positional = np.array([pos_embed(pos, d_model) for pos in range(n_position)])
    positional[:, 0::2] = np.sin(positional[:, 0::2])
    positional[:, 1::2] = np.cos(positional[:, 1::2])
    return torch.FloatTensor(positional)


if __name__ == '__main__':
    a = torch.LongTensor([[1, 2]])
    d = torch.LongTensor([[1, 2, 3]])
    A = torch.randn([1, 2, 512])
    b = Transformers()
    c = b(a, d)
    print(c)
    print(c.size())
