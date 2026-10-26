"""Metric extraction helpers (NMSE / SER / x_NMSE)."""

from typing import Dict, Optional

import torch


SER_KEYS = ("SER_velocity", "SER_langevin", "SER")
X_NMSE_KEYS = ("x_nmse_log", "X_nmse_log", "x_log")


def final_metric_from_log(log_array, snr_len: int) -> torch.Tensor:
    log_tensor = torch.as_tensor(log_array, dtype=torch.float64)
    if log_tensor.numel() == 0:
        return torch.zeros(snr_len, dtype=torch.float64)

    dims = list(log_tensor.shape)
    # Remove leading singleton dimensions to mimic notebook indexing [0,0,0,...].
    while dims and dims[0] == 1:
        log_tensor = log_tensor.squeeze(0)
        dims = list(log_tensor.shape)

    if log_tensor.dim() == 0:
        return log_tensor.unsqueeze(0)

    dims = list(log_tensor.shape)
    if snr_len in dims:
        snr_dim = dims.index(snr_len)
        permute_order = [snr_dim] + [i for i in range(log_tensor.dim()) if i != snr_dim]
        log_tensor = log_tensor.permute(*permute_order)
    else:
        log_tensor = log_tensor.reshape(snr_len, -1)
        return log_tensor[:, -1]

    reshaped = log_tensor.reshape(snr_len, -1)
    return reshaped[:, -1]


def final_nmse_db(oracle_log, snr_len: int) -> torch.Tensor:
    final_values = final_metric_from_log(oracle_log, snr_len)
    final_values = torch.clamp(final_values, min=1e-12)
    return 10.0 * torch.log10(final_values)


def extract_ser(payload: Dict) -> Optional[torch.Tensor]:
    for key in SER_KEYS:
        if key in payload:
            ser = torch.as_tensor(payload[key], dtype=torch.float64)
            return ser.reshape(-1)
    return None


def extract_x_nmse(payload: Dict, snr_len: int) -> Optional[torch.Tensor]:
    for key in X_NMSE_KEYS:
        if key in payload:
            return final_metric_from_log(payload[key], snr_len)
    return None


def _safe_metric_lookup(metric: Optional[torch.Tensor], index: int) -> Optional[float]:
    if metric is None:
        return None
    flat = metric.reshape(-1)
    if flat.numel() == 0:
        return None
    idx = min(index, flat.shape[0] - 1)
    return float(flat[idx].item())
