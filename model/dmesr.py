from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import torchaudio.transforms as T
import torchaudio.functional as F_audio


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DMESRConfig:
    # ── Pre-extracted feature dims (confirmed from inspect_dataset.py) ────
    text_feat_seq_len: int = 50      # token sequence length in pre-extracted text
    text_feat_dim:     int = 768     # BERT hidden dim
    audio_feat_dim:    int = 74      # COVAREP
    video_feat_dim:    int = 35      # FACET

    # ── Encoder output dims ───────────────────────────────────────────────
    text_embed_dim:  int = 768
    audio_embed_dim: int = 256
    video_embed_dim: int = 256

    # ── BiLSTM hidden sizes for audio/video encoders ─────────────────────
    audio_lstm_hidden: int = 128
    video_lstm_hidden: int = 128

    # ── Fusion Transformer ────────────────────────────────────────────────
    fusion_hidden_dim:  int = 512
    fusion_out_dim:     int = 1024
    fusion_num_heads:   int = 4
    fusion_num_layers:  int = 2
    fusion_dropout:     float = 0.1

    # ── Emotion State — MOSEI provides sentiment only ─────────────────────
    emotion_state_dim: int = 1
    emotion_embed_dim: int = 512
    emotion_dims: List[str] = field(default_factory=lambda: [
        "valence", "arousal", "dominance", "confidence", "engagement",
        "uncertainty", "stress", "empathy_needed", "trust", "openness",
        "emotion_strength"
    ])
    label_scale:  float = 3.0        # MOSEI sentiment lives in [-3, 3]

    # ── Temporal (kept for future multi-turn extension) ───────────────────
    temporal_hidden_dim:  int = 256
    temporal_num_layers:  int = 2
    temporal_dropout:     float = 0.1
    memory_max_sentences: int = 64


# ─────────────────────────────────────────────────────────────────────────────
#  Text Encoder  — PrecomputedTextEncoder
#  Input:  (B, 50, 768)  pre-extracted BERT token embeddings
#  Output: (B, 768)      single sentence embedding
# ─────────────────────────────────────────────────────────────────────────────

class PrecomputedTextEncoder(nn.Module):
    """
    Lightweight re-contextualisation of pre-extracted BERT token embeddings.

    A single Transformer encoder layer attends over the 50 token positions,
    then CLS-pool (position 0) gives the sentence representation.
    This costs ~4M extra parameters (all trainable) but gives the fusion
    transformer a richer sentence vector than a naive mean-pool would.

    No pretrained weights are loaded at runtime — the pre-extracted features
    already encode BERT's knowledge; we are only learning a task-specific
    re-weighting on top of them.
    """

    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        # Positional encoding (learnable — 50 positions)
        self.pos_emb = nn.Embedding(cfg.text_feat_seq_len, cfg.text_feat_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_feat_dim,
            nhead=8,
            dim_feedforward=cfg.text_feat_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.proj = nn.Sequential(
            nn.Linear(cfg.text_feat_dim, cfg.text_embed_dim),
            nn.LayerNorm(cfg.text_embed_dim),
            nn.GELU(),
        )

    def forward(self, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feats: (B, 50, 768) — pre-extracted BERT embeddings
        Returns:
            text_embed: (B, 768)
        """
        B, T, D = text_feats.shape
        positions = torch.arange(T, device=text_feats.device).unsqueeze(0)  # (1, T)
        x = text_feats + self.pos_emb(positions)        # (B, T, 768)
        x = self.encoder(x)                             # (B, T, 768)
        cls = x[:, 0, :]                                # CLS-pool
        return self.proj(cls)                           # (B, 768)


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Encoder — BiLSTM over COVAREP feature sequences (T, 74)
# ─────────────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.audio_feat_dim,
            hidden_size=cfg.audio_lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.audio_lstm_hidden * 2, cfg.audio_embed_dim),
            nn.LayerNorm(cfg.audio_embed_dim),
            nn.GELU(),
        )

    def forward(self, audio_feats: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            audio_feats: (B, T=500, 74) — zero-padded COVAREP features
            lengths:     (B,) valid frame counts per sample
        Returns:
            audio_embed: (B, 256)
        """
        out, _ = self.lstm(audio_feats)   # (B, T, 256)
        pooled = self._masked_mean(out, lengths)
        return self.proj(pooled)

    @staticmethod
    def _masked_mean(out: torch.Tensor,
                     lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if lengths is None:
            return out.mean(dim=1)
        mask = (torch.arange(out.shape[1], device=out.device)[None, :]
                < lengths[:, None].to(out.device))          # (B, T)
        mask = mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Video Encoder — BiLSTM over FACET feature sequences (T, 35)
# ─────────────────────────────────────────────────────────────────────────────

class VideoEncoder(nn.Module):
    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.video_feat_dim,
            hidden_size=cfg.video_lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.video_lstm_hidden * 2, cfg.video_embed_dim),
            nn.LayerNorm(cfg.video_embed_dim),
            nn.GELU(),
        )

    def forward(self, video_feats: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.lstm(video_feats)
        pooled = AudioEncoder._masked_mean(out, lengths)
        return self.proj(pooled)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-Modal Attention & Fusion Transformer
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, kv, kv)
        return self.norm(query + self.drop(attended))


class FusionTransformerLayer(nn.Module):
    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        d, h, drop = cfg.fusion_hidden_dim, cfg.fusion_num_heads, cfg.fusion_dropout

        self.sa_t = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.sa_a = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.sa_v = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)

        self.cross_ta = CrossModalAttention(d, h, drop)   # text  ← audio
        self.cross_at = CrossModalAttention(d, h, drop)   # audio ← text
        self.cross_fv = CrossModalAttention(d, h, drop)   # fused ← video

        self.norm_t = nn.LayerNorm(d)
        self.norm_a = nn.LayerNorm(d)
        self.norm_v = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d * 4, d), nn.Dropout(drop),
        )
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, t, a, v):
        t2, _ = self.sa_t(t, t, t);  t = self.norm_t(t + t2)
        a2, _ = self.sa_a(a, a, a);  a = self.norm_a(a + a2)
        v2, _ = self.sa_v(v, v, v);  v = self.norm_v(v + v2)

        t = self.cross_ta(query=t, kv=a)
        a = self.cross_at(query=a, kv=t)

        fused = self.cross_fv(query=(t + a) / 2.0, kv=v)
        fused = self.norm_ffn(fused + self.ffn(fused))
        return fused, t, a


class FusionTransformer(nn.Module):
    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        d = cfg.fusion_hidden_dim

        self.proj_text  = nn.Linear(cfg.text_embed_dim,  d)
        self.proj_audio = nn.Linear(cfg.audio_embed_dim, d)
        self.proj_video = nn.Linear(cfg.video_embed_dim, d)
        self.modal_emb  = nn.Embedding(3, d)

        self.layers = nn.ModuleList([
            FusionTransformerLayer(cfg) for _ in range(cfg.fusion_num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(d, cfg.fusion_out_dim),
            nn.LayerNorm(cfg.fusion_out_dim),
            nn.GELU(),
        )

    def forward(self, text_emb, audio_emb, video_emb):
        t = self.proj_text(text_emb)
        a = self.proj_audio(audio_emb)
        v = self.proj_video(video_emb)

        ids = torch.tensor([0, 1, 2], device=text_emb.device)
        me  = self.modal_emb(ids)
        t, a, v = t + me[0], a + me[1], v + me[2]

        t, a, v = t.unsqueeze(1), a.unsqueeze(1), v.unsqueeze(1)

        fused = None
        for layer in self.layers:
            fused, t, a = layer(t, a, v)

        return self.output_proj(fused.squeeze(1))   # (B, 1024)


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion State Head — 1D sentiment output (compatibility mapped to 11D)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionStateHead(nn.Module):
    """
    Single output: sentiment in [-3, 3]  (tanh * label_scale).
    """

    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        self.label_scale = cfg.label_scale
        self.net = nn.Sequential(
            nn.Linear(cfg.fusion_out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, cfg.emotion_state_dim),  # → 1
        )
        self.dim_names = cfg.emotion_dims

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) sentiment prediction in [-3, 3]."""
        return torch.tanh(self.net(fused)) * self.label_scale

    def to_dict(self, state_vec: torch.Tensor) -> Dict[str, float]:
        """Convert (1,) tensor to 11-D named dict for compatibility."""
        sentiment = float(state_vec[0])

        # Map sentiment (range [-3.0, 3.0]) to 11 continuous VAD + other dimensions in [-1.0, 1.0]
        valence = max(-1.0, min(1.0, sentiment / 3.0))
        arousal = 0.1 + 0.6 * (abs(sentiment) / 3.0)
        dominance = max(-1.0, min(1.0, sentiment / 3.0))
        confidence = max(-1.0, min(1.0, 0.2 + 0.6 * (sentiment / 3.0)))
        engagement = max(-1.0, min(1.0, 0.3 + 0.5 * (abs(sentiment) / 3.0)))
        uncertainty = max(-1.0, min(1.0, 1.0 - (abs(sentiment) / 3.0)))
        stress = max(-1.0, min(1.0, -sentiment / 3.0))
        empathy_needed = max(-1.0, min(1.0, -sentiment / 3.0))
        trust = max(-1.0, min(1.0, 0.1 + 0.6 * (sentiment / 3.0)))
        openness = max(-1.0, min(1.0, 0.1 + 0.6 * (sentiment / 3.0)))
        emotion_strength = max(-1.0, min(1.0, abs(sentiment) / 3.0))

        return {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "confidence": round(confidence, 4),
            "engagement": round(engagement, 4),
            "uncertainty": round(uncertainty, 4),
            "stress": round(stress, 4),
            "empathy_needed": round(empathy_needed, 4),
            "trust": round(trust, 4),
            "openness": round(openness, 4),
            "emotion_strength": round(emotion_strength, 4)
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion Embedding Head — 512D contrastive latent
# ─────────────────────────────────────────────────────────────────────────────

class EmotionEmbedHead(nn.Module):
    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.fusion_out_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, cfg.emotion_embed_dim),
        )

    def forward(self, fused: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        emb = self.net(fused)
        return F.normalize(emb, p=2, dim=-1) if normalize else emb


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Emotion Model (Kept for compatibility/future use)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEmotionModel(nn.Module):
    def __init__(self, cfg: DMESRConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.emotion_embed_dim,
            hidden_size=cfg.temporal_hidden_dim,
            num_layers=cfg.temporal_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.temporal_dropout if cfg.temporal_num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.temporal_hidden_dim * 2, cfg.emotion_embed_dim),
            nn.LayerNorm(cfg.emotion_embed_dim),
        )

    def forward(self, embed_seq: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embed_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(embed_seq)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Full DMESR Model
# ─────────────────────────────────────────────────────────────────────────────

class DMESR(nn.Module):
    def __init__(self, cfg: Optional[DMESRConfig] = None):
        super().__init__()
        self.cfg = cfg or DMESRConfig()

        self.text_encoder  = PrecomputedTextEncoder(self.cfg)
        self.audio_encoder = AudioEncoder(self.cfg)
        self.video_encoder = VideoEncoder(self.cfg)
        self.fusion        = FusionTransformer(self.cfg)
        self.state_head    = EmotionStateHead(self.cfg)
        self.embed_head    = EmotionEmbedHead(self.cfg)
        self.temporal      = TemporalEmotionModel(self.cfg)

    def encode_sentence(
        self,
        text_feats:    torch.Tensor,                   # (B, 50, 768)
        audio_feats:   torch.Tensor,                   # (B, 500, 74)
        video_feats:   torch.Tensor,                   # (B, 500, 35)
        audio_lengths: Optional[torch.Tensor] = None,  # (B,)
        video_lengths: Optional[torch.Tensor] = None,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sentence-level encode pass, aligned with standard API."""
        text_emb  = self.text_encoder(text_feats)
        audio_emb = self.audio_encoder(audio_feats, audio_lengths)
        video_emb = self.video_encoder(video_feats, video_lengths)

        fused = self.fusion(text_emb, audio_emb, video_emb)

        emotion_state = self.state_head(fused)
        emotion_embed = self.embed_head(fused)

        return fused, emotion_state, emotion_embed

    def forward(
        self,
        text_feats:    torch.Tensor,                   # (B, 50, 768)
        audio_feats:   torch.Tensor,                   # (B, 500, 74)
        video_feats:   torch.Tensor,                   # (B, 500, 35)
        audio_lengths: Optional[torch.Tensor] = None,  # (B,)
        video_lengths: Optional[torch.Tensor] = None,  # (B,)
    ) -> Dict[str, torch.Tensor]:

        fused, state, embed = self.encode_sentence(
            text_feats, audio_feats, video_feats, audio_lengths, video_lengths
        )

        return {
            "fused":         fused,
            "emotion_state": state,          # (B, 1)
            "emotion_embed": embed,          # (B, 512)
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveEmotionLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, threshold: float = 0.5):
        super().__init__()
        self.temp = temperature
        self.threshold = threshold

    def forward(self, embeddings: torch.Tensor,
                sentiment: torch.Tensor) -> torch.Tensor:
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        sim = torch.mm(embeddings, embeddings.T) / self.temp  # (B, B)

        with torch.no_grad():
            s = sentiment.view(B)
            dist = (s.unsqueeze(0) - s.unsqueeze(1)).abs()    # (B, B)
            pos_mask = (dist < self.threshold).float()
            pos_mask.fill_diagonal_(0.0)

        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim)
        denom = (exp_sim * (1 - torch.eye(B, device=sim.device))).sum(1, keepdim=True) + 1e-8
        log_prob = sim - torch.log(denom)

        pos_count = pos_mask.sum(1)
        valid = pos_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        loss = -(pos_mask * log_prob).sum(1)[valid] / pos_count[valid]
        return loss.mean()


class DMESRLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.contrastive = ContrastiveEmotionLoss()

    def forward(
        self,
        emotion_state: torch.Tensor,   # (B, 1)
        emotion_embed: torch.Tensor,   # (B, 512)
        target:        torch.Tensor,   # (B, 1)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l_mse = F.mse_loss(emotion_state, target)
        l_con = self.contrastive(emotion_embed, target)
        total = self.alpha * l_mse + self.beta * l_con

        return total, {
            "mse":         float(l_mse),
            "contrastive": float(l_con),
            "total":       float(total),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion Memory Module
# ─────────────────────────────────────────────────────────────────────────────

class EmotionMemoryEntry:
    __slots__ = ("sentence", "timestamp", "emotion_state", "emotion_embedding")

    def __init__(
        self,
        sentence: str,
        timestamp: str,
        emotion_state: Dict[str, float],
        emotion_embedding: torch.Tensor,   # (512,) CPU tensor
    ):
        self.sentence = sentence
        self.timestamp = timestamp
        self.emotion_state = emotion_state
        self.emotion_embedding = emotion_embedding


class EmotionMemory:
    def __init__(self, cfg: DMESRConfig):
        self.max_sentences = cfg.memory_max_sentences
        self.emotion_dims = cfg.emotion_dims
        self._store: List[EmotionMemoryEntry] = []

    def append(self, entry: EmotionMemoryEntry) -> None:
        self._store.append(entry)
        if len(self._store) > self.max_sentences:
            self._store.pop(0)

    def get_history(self, last_k: Optional[int] = None) -> List[EmotionMemoryEntry]:
        if last_k is None:
            return list(self._store)
        return list(self._store[-last_k:])

    def reset(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def compute_velocity(self) -> Optional[Dict[str, float]]:
        if len(self._store) < 2:
            return None
        curr = self._store[-1].emotion_state
        prev = self._store[-2].emotion_state
        return {dim: round(curr[dim] - prev[dim], 4) for dim in self.emotion_dims}

    def compute_acceleration(self) -> Optional[Dict[str, float]]:
        if len(self._store) < 3:
            return None
        curr = self._store[-1].emotion_state
        prev = self._store[-2].emotion_state
        pprev = self._store[-3].emotion_state
        vel_t   = {dim: curr[dim] - prev[dim]  for dim in self.emotion_dims}
        vel_tm1 = {dim: prev[dim] - pprev[dim] for dim in self.emotion_dims}
        return {dim: round(vel_t[dim] - vel_tm1[dim], 4) for dim in self.emotion_dims}

    def summarize(self) -> Dict[str, Dict[str, float]]:
        if not self._store:
            return {}
        states = {dim: [] for dim in self.emotion_dims}
        for entry in self._store:
            for dim in self.emotion_dims:
                states[dim].append(entry.emotion_state[dim])
        summary = {}
        for dim, vals in states.items():
            arr = torch.tensor(vals)
            summary[dim] = {
                "mean": round(float(arr.mean()), 4),
                "std":  round(float(arr.std()), 4) if len(vals) > 1 else 0.0,
                "min":  round(float(arr.min()), 4),
                "max":  round(float(arr.max()), 4)
            }
        return summary

    def get_trend_labels(self, velocity: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        v = velocity or self.compute_velocity()
        if v is None:
            return {}
        labels = {}
        for dim, delta in v.items():
            if delta > 0.05:
                labels[dim] = "Rising ↑"
            elif delta < -0.05:
                labels[dim] = "Falling ↓"
            else:
                labels[dim] = "Stable →"
        return labels

    def to_llm_context(self, last_k: int = 5) -> Dict:
        recent = self.get_history(last_k)
        return {
            "recent_history": [
                {
                    "sentence": e.sentence,
                    "timestamp": e.timestamp,
                    "emotion_state": e.emotion_state,
                }
                for e in recent
            ],
            "conversation_summary": self.summarize(),
            "emotion_velocity": self.compute_velocity(),
            "emotion_acceleration": self.compute_acceleration(),
            "trend_labels": self.get_trend_labels(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Avatar & Voice Parameter Generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_avatar_commands(emotion_state: Dict[str, float]) -> Dict[str, float]:
    v = emotion_state.get("valence", 0.0)
    a = emotion_state.get("arousal", 0.0)
    empathy = emotion_state.get("empathy_needed", 0.0)
    stress = emotion_state.get("stress", 0.0)

    # Derive expression label
    if empathy > 0.6 and v < -0.2:
        expression = "compassion"
    elif v > 0.5 and a > 0.3:
        expression = "joyful"
    elif v < -0.5:
        expression = "concerned"
    elif a > 0.7:
        expression = "alert"
    else:
        expression = "neutral"

    return {
        "expression": expression,
        "eye_contact": float(min(1.0, 0.5 + empathy * 0.5)),
        "smile": float(max(0.0, (v + 1.0) / 2.0 * 0.6)),
        "head_tilt": float(empathy * 0.5),
        "brow_raise": float(max(0.0, a * 0.4)),
        "brow_furrow": float(max(0.0, stress * 0.5)),
    }


def generate_voice_params(emotion_state: Dict[str, float]) -> Dict:
    a = emotion_state.get("arousal", 0.0)
    v = emotion_state.get("valence", 0.0)
    empathy = emotion_state.get("empathy_needed", 0.0)
    stress = emotion_state.get("stress", 0.0)

    # Speed: neutral = 1.0; scale by arousal
    speed = float(max(0.6, min(1.4, 1.0 + (a - 0.0) * 0.4)))

    # Pitch: slight positive shift for upbeat states
    pitch = float(max(0.7, min(1.2, 1.0 + v * 0.15)))

    # Warmth: inverse of stress, boosted by empathy signal
    warmth = float(max(0.3, min(1.0, 0.5 + empathy * 0.4 - stress * 0.2)))

    # Derive emotion label
    if empathy > 0.6:
        emotion = "compassion"
    elif v > 0.5:
        emotion = "warm_positive"
    elif stress > 0.7:
        emotion = "calm_reassuring"
    else:
        emotion = "neutral"

    return {
        "speed": round(speed, 3),
        "pitch": round(pitch, 3),
        "warmth": round(warmth, 3),
        "emotion": emotion,
        "pause_after_sentence_ms": int(200 + empathy * 300),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Stateful Inference Wrapper (Extracts features and drives model)
# ─────────────────────────────────────────────────────────────────────────────

class DMESRInference:
    def __init__(self, model: DMESR, cfg: DMESRConfig, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.memory = EmotionMemory(cfg)

        # Instantiate frozen BERT model for live text feature extraction
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device).eval()
        for p in self.bert.parameters():
            p.requires_grad = False

        self.state_head = model.state_head

    def _extract_text_features(self, text: str) -> torch.Tensor:
        """Tokenize text and extract frozen BERT sequence representations of shape (1, 50, 768)."""
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.text_feat_seq_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state # (1, 50, 768)

    def _extract_audio_features(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC and pitch, interpolate/pad to shape (1, 500, 74) as COVAREP approximation."""
        # Ensure waveform is float32 on correct device
        wave = audio_waveform.to(torch.float32).to(self.device)
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)

        # Check if waveform is practically silent/empty
        if wave.abs().max() < 1e-4:
            return torch.zeros(1, 500, 74, device=self.device)

        # Standard window = 25ms (400 samples), hop = 10ms (160 samples)
        hop_length = 160
        win_length = 400
        mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=24,
            melkwargs={
                "n_fft": 512,
                "n_mels": 26,
                "hop_length": hop_length,
                "win_length": win_length,
            }
        ).to(self.device)

        try:
            mfcc = mfcc_transform(wave).squeeze(0).t() # (num_frames, 24)
            num_frames = mfcc.shape[0]

            # Detect pitch frequency (F0)
            pitch = F_audio.detect_pitch_frequency(
                wave,
                sample_rate=16000,
                frame_time=0.01
            ).squeeze(0) # (pitch_frames,)

            # Align pitch and mfcc shapes
            p_len = len(pitch)
            if p_len < num_frames:
                pitch = torch.cat([pitch, torch.zeros(num_frames - p_len, device=self.device)])
            else:
                pitch = pitch[:num_frames]

            # Fill COVAREP approximate slots
            features = torch.zeros(num_frames, 74, device=self.device)
            features[:, :24] = mfcc
            features[:, 24] = pitch

            # Interpolate to exactly 500 frames along sequence dim
            features = features.unsqueeze(0).transpose(1, 2) # (1, 74, num_frames)
            if num_frames != 500:
                features = F.interpolate(features, size=500, mode="linear", align_corners=False)
            features = features.transpose(1, 2) # (1, 500, 74)
            return features
        except Exception:
            return torch.zeros(1, 500, 74, device=self.device)

    def _extract_video_features(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Represent FACET features as neutral (zero tensor of shape (1, 500, 35))."""
        return torch.zeros(1, 500, 35, device=self.device)

    @torch.no_grad()
    def process_sentence(
        self,
        text: str,
        audio_waveform: torch.Tensor,    # (T_audio,)
        video_frames: torch.Tensor,       # (3, 8, 112, 112)
        timestamp: str = "",
    ) -> Dict:
        """Process a single sentence and return the full emotion output package."""
        # 1. Feature extraction
        text_feats = self._extract_text_features(text)
        audio_feats = self._extract_audio_features(audio_waveform)
        video_feats = self._extract_video_features(video_frames)

        # 2. Model forward pass
        _, state_vec, embed_vec = self.model.encode_sentence(
            text_feats, audio_feats, video_feats
        )

        state_dict = self.state_head.to_dict(state_vec[0])
        embed_cpu  = embed_vec[0].cpu()

        # 3. Update memory
        entry = EmotionMemoryEntry(
            sentence=text,
            timestamp=timestamp,
            emotion_state=state_dict,
            emotion_embedding=embed_cpu,
        )
        self.memory.append(entry)

        # 4. Derive outputs
        avatar_cmds  = generate_avatar_commands(state_dict)
        voice_params = generate_voice_params(state_dict)
        llm_context  = self.memory.to_llm_context(last_k=5)

        return {
            "sentence":        text,
            "timestamp":       timestamp,
            "emotion_state":   state_dict,
            "emotion_embedding": embed_cpu.tolist(),
            "emotion_velocity": self.memory.compute_velocity(),
            "trend_labels":    self.memory.get_trend_labels(),
            "avatar_commands": avatar_cmds,
            "voice_params":    voice_params,
            "llm_context":     llm_context,
        }

    def reset_conversation(self) -> None:
        self.memory.reset()