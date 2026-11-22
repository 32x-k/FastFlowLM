# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Portions of this file consist of AI-generated content.

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
import torch


class Plamo3(__Q4NX_Converter, model_arch=ModelArch.PLAMO3):
    """PLaMo-3 converter.

    PLaMo-3 differs from the per-tensor name_map flow in three ways, all handled
    here:

    * fused ``blk.N.attn_qkv.weight`` -- split into q/k/v, then each KV head is
      replicated so the model always presents 8 KV heads to the engines (the
      prefill attention and decoding-layer kernels have no GQA 6:1 or MQA slot).
    * fused ``blk.N.ffn_up.weight`` -- the first ``intermediate_size`` output
      rows are the gate (SiLU branch), the second half is the up projection.
    * norm tensors carry no ".weight" suffix on the post norms, and every norm
      uses Plamo3RMSNorm's ``(offset + weight) * x`` form. The offset is folded
      into the stored weight here.
    """

    # GGUF norm tensors already store the effective weight (plain multiply,
    # no 1+w offset): llama.cpp applies them directly via build_norm().
    _NORM_OFFSETS = {
        "input_layernorm.weight": 0.0,
        "post_attention_layernorm.weight": 0.0,
        "pre_feedforward_layernorm.weight": 0.0,
        "post_feedforward_layernorm.weight": 0.0,
        "q_norm.weight": 0.0,
        "k_norm.weight": 0.0,
        "norm.weight": 0.0,
    }

    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()
        self.num_kv_heads = self._gguf_int("plamo3.attention.head_count_kv")
        self.num_q_heads = self._gguf_int("plamo3.attention.head_count")
        self.head_dim = self._gguf_int("plamo3.attention.key_length")
        if self.num_kv_heads is None:
            raise ValueError("plamo3.attention.head_count_kv missing from GGUF")
        if 8 % self.num_kv_heads != 0:
            raise ValueError(f"cannot replicate {self.num_kv_heads} KV heads to 8")

    def _gguf_int(self, key):
        field = self.gguf_reader.fields.get(key)
        return int(field.contents()) if field is not None else None

    def _requant_pack(self, w: torch.Tensor) -> torch.Tensor:
        """bf16/f32 (out, in) weight -> Q4_1 -> q4nx block."""
        data = quantize(w.to(torch.float32).contiguous().numpy(), self.default_tensor_type).copy()
        d, m, qw = GGUFTensor.unpack_q4_1(data, w.shape[1])
        return self._pack_q4nx(d, m, qw)

    def _emit_norm(self, gguf_name: str, q4nx_name: str):
        t = self.gguf_tensors[gguf_name]
        offset = next(
            v for k, v in sorted(self._NORM_OFFSETS.items(), key=lambda kv: -len(kv[0]))
            if q4nx_name.endswith(k)
        )
        w = t.dequantize().to(torch.float32) + offset
        self.q4nx_tensors[q4nx_name] = w.to(torch.bfloat16)

    def _emit_split_qkv(self, bid: int):
        t = self.gguf_tensors[f"blk.{bid}.attn_qkv.weight"]
        w = t.dequantize()  # (q_out + kv_out + kv_out, hidden)
        q_out = self.num_q_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        assert w.shape[0] == q_out + 2 * kv_out, f"unexpected fused qkv shape {w.shape}"
        q, k, v = w[:q_out], w[q_out:q_out + kv_out], w[q_out + kv_out:]

        # replicate each KV head so num_kv_heads becomes 4 (adjacent pairs);
        # the decode kernel has 4 attention core pairs, one KV head each
        rep = 4 // self.num_kv_heads
        if rep > 1:
            k = k.view(self.num_kv_heads, self.head_dim, w.shape[1]) \
                 .repeat_interleave(rep, dim=0).reshape(-1, w.shape[1])
            v = v.view(self.num_kv_heads, self.head_dim, w.shape[1]) \
                 .repeat_interleave(rep, dim=0).reshape(-1, w.shape[1])

        base = f"model.layers.{bid}.self_attn"
        self.q4nx_tensors[f"{base}.q_proj.weight"] = self._requant_pack(q)
        self.q4nx_tensors[f"{base}.k_proj.weight"] = self._requant_pack(k)
        self.q4nx_tensors[f"{base}.v_proj.weight"] = self._requant_pack(v)

    def _emit_split_ffn_up(self, bid: int):
        t = self.gguf_tensors[f"blk.{bid}.ffn_up.weight"]
        w = t.dequantize()  # (2 * intermediate, hidden): gate rows first
        gate, up = w[:self.intermediate_size], w[self.intermediate_size:]
        base = f"model.layers.{bid}.mlp"
        self.q4nx_tensors[f"{base}.gate_proj.weight"] = self._requant_pack(gate)
        self.q4nx_tensors[f"{base}.up_proj.weight"] = self._requant_pack(up)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != 'language':
            raise ValueError(f"Unsupported weights_type: {weights_type} for Plamo3 model")
        self.q4nx_tensors = {}

        for key, gguf_tensor in self.gguf_tensors.items():
            name = gguf_tensor.name

            if name == "token_embd.weight":
                # plamo-3-translate configs set scale_embedding=false; emit the
                # embedding as plain bf16 (no sqrt(D) fold).
                w = gguf_tensor.dequantize().to(torch.float32)
                self.q4nx_tensors[self.forward_name_map[name]] = w.to(torch.bfloat16)
                continue

            if "attn_qkv" in name or "ffn_up" in name:
                continue  # fused tensors are expanded below

            if "norm" in name:
                self._emit_norm(name, self.forward_name_map[name])
                continue

            if gguf_tensor.tensor_type == GGMLQuantizationType.Q4_K:
                # unpack_q4_k returns (t, u, q) where w = t*q - u (u is subtracted),
                # whereas _pack_q4nx stores Q4_1 w = d*q + m (m is added).
                t, u, q = gguf_tensor.unpack_q4_k(gguf_tensor.data, gguf_tensor.shape[0])
                self.q4nx_tensors[self.forward_name_map[name]] = self._pack_q4nx(t, -u, q)
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            self.q4nx_tensors[self.forward_name_map[name]] = self._pack_q4nx(*unpacked)

        for bid in range(self.num_layers):
            self._emit_split_qkv(bid)
            self._emit_split_ffn_up(bid)

        self._export_q4nx_tensors(q4nx_path)
