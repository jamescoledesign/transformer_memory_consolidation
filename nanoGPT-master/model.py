"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.utils.generic import ModelOutput

@dataclass
class TransfoXLModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    logits: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ffn_activations: Optional[Tuple[torch.FloatTensor]] = None
    grid_activations: Optional[Tuple[torch.FloatTensor]] = None
    rnn_hidden: Optional[Tuple[torch.FloatTensor]] = None


class NMDA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(NMDA, self).__init__()
        self.alpha = alpha
        if alpha <= 0:
            self.a = None
        else:
            self.a = math.log(self.alpha)
        self.beta = beta

    def forward(self, x):
        if self.a is None:
            return x
        else:
            return x * torch.sigmoid(self.beta * x - self.a)


_act_fns = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(inplace=True),
    "swish": nn.SiLU(),
    "linear": nn.Identity(),
}


class SinusoidPosEmbedding(nn.Module):
    def __init__(self, demb):
        super(SinusoidPosEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner,
        dropout,
        pre_lnorm=False,
        ffn_act_ftn="gelu",
        alpha=1.0,
        beta=1.0,
    ):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        if ffn_act_ftn != "nmda":
            act_nn = _act_fns[ffn_act_ftn]
        else:
            act_nn = NMDA(alpha, beta)

        self.fc1 = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            act_nn,
        )
        self.fc2 = nn.Sequential(
            nn.Linear(d_inner, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            ffn_hid = self.fc1(self.layer_norm(inp))
            core_out = self.fc2(ffn_hid)
            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            ffn_hid = self.fc1(inp)
            core_out = self.fc2(ffn_hid)
            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return [output, ffn_hid]


class MultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=True, **kwargs
    ):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum("ibnd,jbnd->ijbn", (head_q, head_k))
        attn_score.mul_(self.scale)

        mask_value = torch.finfo(attn_score.dtype).min

        if attn_mask is not None and attn_mask.any().item():
            attn_mask = attn_mask == 1
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], mask_value)
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], mask_value)
                    .type_as(attn_score)
                )

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            outputs = [h + attn_out]
        else:
            ##### residual connection + layer normalization
            outputs = [self.layer_norm(h + attn_out)]
        outputs.append(attn_prob)

        return outputs


class RelMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        tgt_len=None,
        ext_len=None,
        mem_len=None,
        pre_lnorm=False,
        **kwargs,
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(
            qlen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum(
            "ibnd,jnd->ijbn", (rr_head_q, r_head_k)
        )  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        mask_value = torch.finfo(attn_score.dtype).min

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            attn_mask = attn_mask == 1
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], mask_value)
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], mask_value)
                    .type_as(attn_score)
                )

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            outputs = [w + attn_out]
        else:
            ##### residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]
        outputs.append(attn_prob)

        return outputs


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head
        B_ = torch.einsum(
            "ibnd,jnd->ijbn", (w_head_q, r_emb)
        )  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        mask_value = torch.finfo(attn_score.dtype).min

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            attn_mask = attn_mask == 1
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], mask_value)
                    .type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], mask_value)
                    .type_as(attn_score)
                )

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            outputs = [w + attn_out]
        else:
            ##### residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]
        outputs.append(attn_prob)

        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=kwargs.get("pre_lnorm"),
            ffn_act_ftn=kwargs.get("ffn_act_ftn"),
            alpha=kwargs.get("alpha"),
            beta=kwargs.get("beta"),
        )

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        attn_outputs = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        ff_output, core_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output, core_output] + attn_outputs[1:]
        return outputs


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=kwargs.get("pre_lnorm"),
            ffn_act_ftn=kwargs.get("ffn_act_ftn"),
            alpha=kwargs.get("alpha"),
            beta=kwargs.get("beta"),
        )

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        attn_outputs = self.dec_attn(
            dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        ff_output, core_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output, core_output] + attn_outputs[1:]
        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=kwargs.get("pre_lnorm"),
            ffn_act_ftn=kwargs.get("ffn_act_ftn"),
            alpha=kwargs.get("alpha"),
            beta=kwargs.get("beta"),
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        attn_outputs = self.dec_attn(
            dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        ff_output, core_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output, core_output] + attn_outputs[1:]
        return outputs


class TransfoXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_inner = config.d_inner

        self.drop = nn.Dropout(config.dropout)

        self.n_layer = config.n_layer

        self.mem_len = config.mem_len
        self.tgt_len = config.tgt_len
        self.ext_len = config.ext_len
        self.max_klen = self.tgt_len + self.ext_len + self.mem_len

        self.attn_type = config.attn_type

        self.word_emb = nn.Embedding(config.vocab_size + 1, config.d_embed)
        self.grid_emb = nn.Sequential(
            nn.Linear(config.d_rnn, config.d_embed), nn.ReLU(inplace=True)
        )
        self.rew_emb = nn.Embedding(2, config.d_embed)
        self.valid_seq_emb = nn.Embedding(3, config.d_embed)
        self.ln_f = nn.LayerNorm(config.d_model)

        self.layers = nn.ModuleList()
        if self.attn_type == 0:  # the default attention
            for _ in range(self.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        self.n_head,
                        self.d_model,
                        self.d_head,
                        self.d_inner,
                        config.dropout,
                        tgt_len=self.tgt_len,
                        ext_len=self.ext_len,
                        mem_len=self.mem_len,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        ffn_act_ftn=config.ffn_act_ftn,
                        alpha=config.alpha,
                        beta=config.beta,
                    )
                )
        elif self.attn_type == 1:  # learnable embeddings
            for _ in range(self.n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        self.n_head,
                        self.d_model,
                        self.d_head,
                        self.d_inner,
                        config.dropout,
                        tgt_len=self.tgt_len,
                        ext_len=self.ext_len,
                        mem_len=self.mem_len,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        ffn_act_ftn=config.ffn_act_ftn,
                        alpha=config.alpha,
                        beta=config.beta,
                    )
                )
        elif self.attn_type in [2, 3]:  # absolute embeddings
            for _ in range(self.n_layer):
                self.layers.append(
                    DecoderLayer(
                        self.n_head,
                        self.d_model,
                        self.d_head,
                        self.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        ffn_act_ftn=config.ffn_act_ftn,
                        alpha=config.alpha,
                        beta=config.beta,
                    )
                )

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len
        self._create_params()

        # Initialize weights and apply final processing
        self.init_weights()

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = SinusoidPosEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head)
            )
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.n_head, self.d_head)
            )
            self.r_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head)
            )
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = SinusoidPosEmbedding(self.d_model)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weight(self, weight):
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(
        self,
        grid_seq: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        exclude_last: Optional[bool] = True,
        rew_seq: Optional[torch.LongTensor] = None,
        val_seq: Optional[torch.LongTensor] = None,
    ) -> TransfoXLModelOutput:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if mems is None:
            mems = self.init_mems()

        if exclude_last:
            cls_token = torch.ones_like(input_ids[:, :1]) * self.n_token
            input_ids = torch.cat([input_ids, cls_token], dim=1)

        bsz, qlen = input_ids.size()
        obs_emb = self.word_emb(input_ids)
        grid_seq = self.grid_emb(grid_seq)
        _grid_seq = grid_seq.transpose(0, 1).contiguous()
        if not exclude_last:
            rew_emb = self.rew_emb(rew_seq)
            seq_emb = self.valid_seq_emb(val_seq)
            obs_emb = obs_emb + rew_emb + seq_emb
        obs_emb = obs_emb.transpose(0, 1).contiguous()
        word_emb = torch.cat([_grid_seq, obs_emb], dim=-1)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (
                torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)
            ).byte()[
                :, :, None
            ]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen
            ).byte()[:, :, None]

        hids = []
        ffn_acts = [] if output_attentions else None
        attentions = [] if output_attentions else None
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(
                klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype
            )
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(
                    core_out,
                    pos_emb,
                    self.r_w_bias,
                    self.r_r_bias,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                )
                core_out = layer_outputs[0]
                hids.append(core_out)
                if output_attentions:
                    ffn_acts.append(layer_outputs[1])
                    attentions.append(layer_outputs[2])
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(
                    core_out,
                    r_emb,
                    self.r_w_bias[i],
                    r_bias,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                )
                core_out = layer_outputs[0]
                hids.append(core_out)
                if output_attentions:
                    ffn_acts.append(layer_outputs[1])
                    attentions.append(layer_outputs[2])
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(
                klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype
            )
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                layer_outputs = layer(
                    core_out, dec_attn_mask=dec_attn_mask, mems=mems_i
                )
                core_out = layer_outputs[0]
                hids.append(core_out)
                if output_attentions:
                    ffn_acts.append(layer_outputs[1])
                    attentions.append(layer_outputs[2])
        elif self.attn_type == 3:  # no position encoding absolute
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                layer_outputs = layer(
                    core_out, dec_attn_mask=dec_attn_mask, mems=mems_i
                )
                core_out = layer_outputs[0]
                hids.append(core_out)
                if output_attentions:
                    ffn_acts.append(layer_outputs[1])
                    attentions.append(layer_outputs[2])

        if exclude_last:
            qlen = qlen - 1
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        if output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            ffn_acts = tuple(t.transpose(0, 1).contiguous() for t in ffn_acts)
        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = self.ln_f(core_out)
        core_out = core_out.transpose(0, 1).contiguous()

        return TransfoXLModelOutput(
            last_hidden_state=core_out,
            mems=new_mems,
            attentions=attentions,
            ffn_activations=ffn_acts,
            grid_activations=grid_seq,
        )


class xlTEM(nn.Module):
    def __init__(self, config):
        super(xlTEM, self).__init__()
        self.transformer = TransfoXL(config)
        self.rnn = RNN(config)
        self.classifier = nn.Linear(config.d_model, config.vocab_size)
        nn.init.normal_(self.classifier.weight, 0.0, config.init_std)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, act, obs, prev_hidden, mems):
        seq_len = act.shape[1]
        g = [prev_hidden]
        for n in range(seq_len):
            prev_hidden = self.rnn(act[:, n], prev_hidden)
            g.append(prev_hidden)
        g = torch.stack(g, dim=1)
        outputs = self.transformer(g, obs, mems)
        hidden = outputs.last_hidden_state[:, -1]
        outputs.logits = self.classifier(hidden)
        outputs.rnn_hidden = prev_hidden.detach()
        return outputs


_act_ftns = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.d_rnn = config.d_rnn
        self.n_a = config.n_a
        self.side_len = config.side_len
        Wa = torch.zeros(self.n_a, self.d_rnn, self.d_rnn).float()
        self.weight = nn.Parameter(Wa)
        self.act_ftn = _act_ftns[config.rnn_act_ftn]

    def forward(self, a, prev_hidden):
        out = self.act_ftn(
            prev_hidden + torch.einsum("bi,bij->bj", prev_hidden, self.weight[a])
        )
        return out

    def init_hidden(self, pos):
        return torch.randn((pos.shape[0], self.d_rnn)).to(pos.device) / math.sqrt(
            self.d_rnn
        )

# ==============================================================
    
# Code from nanoGPT below 

# ==============================================================


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        # self.nmda    = NMDA()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        # x = self.nmda(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    