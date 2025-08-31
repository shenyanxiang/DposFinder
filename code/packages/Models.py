import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from packages.local_ctd import CalculateCTD
from packages.local_AAComposition import CalculateDipeptideComposition
from packages.modules import TransformerLayer, MLPLayer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np


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
    def __init__(self, hyp_params, unfreeze_last=False, return_subseq=False):

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

        self.layers = nn.ModuleList(
            [
                TransformerLayer(self.hid_dim, self.heads, self.dropout_rate, self.attn_dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.return_subseq = return_subseq

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
        esm_emb =  torch.cat([emb[i, :len(strs[i]) + 1].mean(0).unsqueeze(0)
                        for i in range(batch_size)], dim=0) # average pooling along the sequence
        emb = emb * padding_mask.unsqueeze(-1).type_as(emb)

        emb = emb.transpose(1, 2)
        emb = self.conv(emb)
        emb = emb.transpose(1, 2)

        for layer in self.layers:
            emb, attn = layer(
                emb, mask=padding_mask.unsqueeze(1).unsqueeze(2)
            )
        if self.return_subseq:
            index_list = []
            for i in range(batch_size):
                mean_attn = attn[i, :, :len(strs[i]), :len(strs[i])].sum(0).mean(0).cpu().numpy()
                mean_attn = np.log(mean_attn)
                norm_attn = (mean_attn-min(mean_attn))/(max(mean_attn)-min(mean_attn))
                max_sum_index = np.argmax(np.convolve(norm_attn, np.ones(150), mode='valid'))

                # ##calculate key position percent###
                # key_aa_num = np.sum(norm_attn>0.5)
                # key_aa_num_region = np.sum(norm_attn[max_sum_index:max_sum_index+250]>0.5)
                # index_start = key_aa_num_region/key_aa_num

                # ##calculate average region attention score
                # region_attn = norm_attn[max_sum_index:max_sum_index+50].sum()
                # index_start = region_attn/50

                ##return the whole attn score###
                # index_start = norm_attn

                index_start = max_sum_index
                index_list.append(index_start)
        else:
            out = torch.cat([emb[i, :len(strs[i]) + 1].mean(0).unsqueeze(0)
                            for i in range(batch_size)], dim=0) # average pooling along the sequence
        
        if self.return_embedding:
            return out
        elif self.return_subseq:
            return index_list
        else:
            logits = self.clf(out)
            if self.return_attn:
                return logits, attn
            else:
                return logits
            
class SPIKEHUNTER(nn.Module):
    def __init__(self, hyp_params,
                 unfreeze_last=False, n_hidden = 568,
                 dropout_rate=0.0,
                 return_embedding=False):

        super().__init__()
        self.pretrained_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.repr_layer = 33
        self.net = nn.Sequential(*[
        nn.Linear(1280, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, 128),
        nn.GELU(),
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
        
class DEPOSCOPE(nn.Module):
    def __init__(self, hyp_params):
        super().__init__()
        max_length = 1024
        self.pretrained_model, _ = esm.pretrained.esm2_t12_35M_UR50D()
        self.max_length = max_length
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1)
        conv_length = max_length - 2 * (5 - 1)
        self.fc1 = nn.Linear(128 * conv_length, 32)
        self.classifier = nn.Linear(32, 1)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if name.startswith(f"pretrained_model.layers.{11}"):
                param.requires_grad = True

    def pad_or_truncate(self, tensor, target_len):
        # tensor: (batch_size, seq_len, feature_dim) or (batch_size, seq_len)
        seq_len = tensor.shape[1]
        if seq_len < target_len:
            pad_shape = list(tensor.shape)
            pad_shape[1] = target_len - seq_len
            pad_tensor = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_tensor], dim=1)
        elif seq_len > target_len:
            tensor = tensor[:, :target_len]
        return tensor

    def forward(self, strs, toks):
        # toks: (batch_size, seq_len)
        batch_size, seq_len = toks.shape

        out = self.pretrained_model(toks, repr_layers=[], return_contacts=False)
        token_logits = out['logits']  # (batch_size, seq_len, vocab_size)
        feature, _ = token_logits.max(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)

        feature = self.pad_or_truncate(feature, self.max_length)  # (batch_size, max_length, 1)
        x = feature.squeeze(-1).unsqueeze(1).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        logits = self.classifier(x)
        return logits

class PHAGEDPO(nn.Module):
    def __init__(self, hyp_params):
        super().__init__()
        ctd_dim = 147
        dipep_dim = 400
        self.feature_dim = 1 + 1 + 1 + 20 + 3 + ctd_dim + dipep_dim

        k_features = getattr(hyp_params, 'k_features', self.feature_dim)
        if isinstance(k_features, str) and k_features == 'all':
            k_features = 'all'
        else:
            k_features = min(int(k_features), self.feature_dim)

        self.k_features = k_features
        self.C = getattr(hyp_params, 'svm_C', 1.0)
        self.kernel = getattr(hyp_params, 'svm_kernel', 'rbf')
        self.gamma = getattr(hyp_params, 'svm_gamma', 'scale')

        self.model = Pipeline([
            ('scl', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif, k=self.k_features)),
            ('clf', SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True))
        ])

    @staticmethod
    def extract_features(seq):
        pa = ProteinAnalysis(seq)
        features = []
        features.append(len(seq))
        features.append(pa.aromaticity())
        features.append(pa.isoelectric_point())
        aa_order = 'GALVIPFSTCYNQDERKHWM'
        aa_percent = pa.get_amino_acids_percent()
        features.extend([aa_percent.get(aa, 0.0) for aa in aa_order])
        helix, turn, sheet = pa.secondary_structure_fraction()
        features.extend([helix, turn, sheet])
        ctd = CalculateCTD(seq)
        features.extend(list(ctd.values()))
        dipep = CalculateDipeptideComposition(seq)
        features.extend(list(dipep.values()))
        return np.array(features, dtype=np.float32)

    def batch_extract(self, seqs):
        feats = [self.extract_features(seq) for seq in seqs]
        feats = np.stack(feats, axis=0)
        if feats.shape[1] != self.feature_dim:
            raise ValueError(f"num of features inconsistent, in fact {feats.shape[1]}，expected {self.feature_dim}")
        return feats

    def fit(self, strs, y):
        X = self.batch_extract(strs)
        self.model.fit(X, np.array(y))

    def forward(self, strs, toks=None):
        X = self.batch_extract(strs)
        probs = self.model.predict_proba(X)[:, 1]
        return torch.tensor(probs, dtype=torch.float32)

    def predict(self, strs, toks=None):
        X = self.batch_extract(strs)
        return self.model.predict(X)
        
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
        device = torch.device('cuda')
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