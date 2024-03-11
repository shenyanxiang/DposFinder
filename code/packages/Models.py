import torch
import torch.nn as nn
import esm
from packages.modules import TransformerLayer, MLPLayer


class ESM2FINETUNE(nn.Module):
    def __init__(self, hyp_params,emb_dim=1280, repr_layer=33,
                 unfreeze_last=True, hid_dim=256,
                 dropout_rate=0.4,
                 return_embedding=False):

        super().__init__()
        self.pretrained_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.repr_layer = repr_layer
        self.clf = MLPLayer(in_dim=emb_dim, hid_dim=hid_dim,
                            dropout_rate=dropout_rate)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        if unfreeze_last:
            for name, param in self.named_parameters():
                if name.startswith(f"pretrained_model.layers.{self.repr_layer-1}"):
                    param.requires_grad = True

        self.return_embedding = return_embedding

    def forward(self, strs, toks):
        batch = toks.shape[0]
        out = self.pretrained_model(toks, repr_layers=[
                                    self.repr_layer], return_contacts=False)  # (bs, seq_len, emb_dim)
        emb = torch.cat([out["representations"][33][i, 1: len(
            strs[i]) + 1].mean(0).unsqueeze(0) for i in range(batch)], dim=0)
        if self.return_embedding:
            return emb
        else:
            logits = self.clf(emb)
            return logits

class DPOSFINDER(nn.Module):
    def __init__(self, hyp_params, unfreeze_last=False):

        super().__init__()
        self.emb_dim = hyp_params.emb_dim
        self.repr_layer = 33
        self.pretrained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.padding_idx = alphabet.padding_idx
        self.hid_dim = hyp_params.hid_dim
        self.conv = nn.Conv1d(self.emb_dim, self.hid_dim, 1, 1, bias=False)
        self.num_layers = hyp_params.n_layers
        self.dropout_rate = hyp_params.embed_dropout
        self.return_embedding = hyp_params.return_embedding
        self.return_attn = hyp_params.return_attn
        self.attn_dropout = hyp_params.attn_dropout
        self.heads = hyp_params.num_heads
        self.clf = nn.Linear(self.hid_dim, 1)
        self.net = nn.Sequential(*[
            nn.Linear(1280, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 128),
            nn.ReLU(),
            nn.Linear(128,1)])

        self.layers = nn.ModuleList(
            [
                TransformerLayer(self.hid_dim, self.heads, self.dropout_rate, self.attn_dropout)
                for _ in range(self.num_layers)
            ]
        )

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        if unfreeze_last:
            for name, param in self.named_parameters():
                if name.startswith(f"pretrained_model.layers.{self.repr_layer-1}"):
                    param.requires_grad = True

    def forward(self, strs, toks):
        batch_size = toks.shape[0]
        padding_mask = (toks != self.padding_idx)[:,1:-1]
        out = self.pretrained_model(
            toks, repr_layers=[self.repr_layer], return_contacts=False)
        emb = out["representations"][33][:, 1:-1, :] #bs,seq_len,emb_dim
        # out2 = torch.cat([emb[i, :len(strs[i]) + 1].mean(0).unsqueeze(0)
        #                 for i in range(batch_size)], dim=0) # average pooling along the sequence
        emb = emb * padding_mask.unsqueeze(-1).type_as(emb)

        emb = emb.transpose(1, 2)
        emb = self.conv(emb)
        emb = emb.transpose(1, 2)

        for layer in self.layers:
            emb, attn = layer(
                emb, mask=padding_mask.unsqueeze(1).unsqueeze(2)
            )
        out = torch.cat([emb[i, :len(strs[i]) + 1].mean(0).unsqueeze(0)
                        for i in range(batch_size)], dim=0) # average pooling along the sequence
        
        if self.return_embedding:
            return out
        else:
            logits = self.clf(out)
            if self.return_attn:
                return logits, attn
            else:
                return logits
            
class SPIKEHUNTER(nn.Module):
    def __init__(self, hyp_params,
                 unfreeze_last=True, n_hidden = 568,
                 dropout_rate=0.4,
                 return_embedding=False):

        super().__init__()
        self.pretrained_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.repr_layer = 33
        self.net = nn.Sequential(*[
        nn.Linear(1280, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 128),
        nn.ReLU(),
        nn.Linear(128,1)])

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        if unfreeze_last:
            for name, param in self.named_parameters():
                if name.startswith(f"pretrained_model.layers.{self.repr_layer-1}"):
                    param.requires_grad = True

        self.return_embedding = return_embedding

    def forward(self, strs, toks):
        toks = toks[:, :1022]
        batch = toks.shape[0]
        out = self.pretrained_model(toks, repr_layers=[
                                    self.repr_layer], return_contacts=False)  # (bs, seq_len, emb_dim)
        emb = torch.cat([out["representations"][33][i, 1: len(
            strs[i]) + 1].mean(0).unsqueeze(0) for i in range(batch)], dim=0)
        if self.return_embedding:
            return emb
        else:
            logits = self.net(emb)
            return logits
        
class TAPETRANSFORMER(nn.Module):
    def __init__(self, hyp_params,emb_dim=768, hid_dim=256,
                 dropout_rate=0.4,
                 return_embedding=False):

        super().__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.conv = nn.Conv1d(self.emb_dim, self.hid_dim, 1, 1, bias=False)
        self.num_layers = hyp_params.n_layers
        self.dropout_rate = hyp_params.embed_dropout
        self.return_embedding = hyp_params.return_embedding
        self.return_attn = hyp_params.return_attn
        self.attn_dropout = hyp_params.attn_dropout
        self.heads = hyp_params.num_heads
        self.return_embedding = return_embedding
        self.layers = nn.ModuleList(
            [
                TransformerLayer(self.hid_dim, self.heads, self.dropout_rate, self.attn_dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.clf = nn.Linear(self.hid_dim, 1)

    def forward(self, strs, toks):
        batch_size = len(strs)
        
        emb = toks
        device = torch.device('cuda:1')
        emb = emb.to(device)
        emb = emb.transpose(1, 2)
        emb = self.conv(emb)
        emb = emb.transpose(1, 2)

        for layer in self.layers:
            emb, attn = layer(
                emb
            )
        out = torch.cat([emb[i, :].mean(0).unsqueeze(0)
                        for i in range(batch_size)], dim=0) # average pooling along the sequence
        
        if self.return_embedding:
            return out
        else:
            logits = self.clf(out)
            if self.return_attn:
                return logits, attn
            else:
                return logits