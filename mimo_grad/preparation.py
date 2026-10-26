"""Channel, pilot and symbol state preparation for MIMO inference."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def select_snr_parameters(
    snr_value: float,
    test_config,
    num_classes: int,
) -> Tuple[float, np.ndarray, float]:
    """Return Langevin schedule parameters for a given SNR value."""
    if snr_value < test_config.low_snr_threshold:
        params = test_config.low_snr_params
    else:
        params = test_config.high_snr_params

    temp_x = params.temp_x
    sigmas_x = np.linspace(
        params.sigma_x_range[0], params.sigma_x_range[1], num_classes
    )
    epsilon = params.epsilon
    return temp_x, sigmas_x, epsilon


def initialize_pilot_state(
    pilots_conj: torch.Tensor,
    H_herm_complex: torch.Tensor,
    local_noise: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate noisy pilot measurements and a fresh channel sample."""
    y_pilots = torch.matmul(pilots_conj, H_herm_complex)
    y_pilots = y_pilots + np.sqrt(local_noise) * torch.randn_like(y_pilots)
    H_current = torch.randn_like(H_herm_complex)
    oracle = H_herm_complex
    return y_pilots, H_current, oracle


def initialize_symbol_state(
    H_test_real_repr: torch.Tensor,
    generator,
    num_channels: int,
    NR: int,
    NT: int,
    batch_size_x: int,
    local_noise: float,
    device: torch.device | str,
    y_pilots: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data structures for the symbol update step."""
    x_current = ((1 + 1) * torch.rand(num_channels, 2 * NT, batch_size_x) + 1).to(
        device=device
    )
    indices = generator.random_indices(NT, batch_size_x * num_channels)
    j_indices = generator.joint_indices(indices)
    x_true = generator.modulate(indices)
    x_true = torch.reshape(x_true, (num_channels, batch_size_x, 2 * NT))
    x_true = torch.transpose(x_true, -1, -2)
    y_x = torch.matmul(H_test_real_repr.double(), x_true.double()).to(device=device).float()
    y_x = y_x + np.sqrt(local_noise) * torch.randn_like(y_x).to(device=device)

    H_current_x = torch.zeros([num_channels, 2 * NR, 2 * NT])

    y_x_complex = y_x.chunk(2, dim=1)
    y_x_complex = (y_x_complex[0] - 1j * y_x_complex[1])
    y_x_complex = torch.transpose(y_x_complex, -1, -2)
    y_H = torch.cat((y_pilots.to(device=device), y_x_complex.to(device=device)), dim=1)

    return x_current, j_indices, y_x, H_current_x, y_H


def fill_real_channel_representation(
    H_complex: torch.Tensor,
    real_buffer: torch.Tensor,
    NR: int,
    NT: int,
    device: torch.device | str,
) -> None:
    """Populate the alternative real-valued block matrix in place.

    Real-imag block layout after the bottom-left sign flip:
        [[ +real(H^H), +imag(H^H) ],
         [ -imag(H^H), +real(H^H) ]]
    where ``H^H = transpose(conj(H_complex))``.

    ``gradients.get_channel_real_representation`` uses the different convention
    ``[[+R, -I], [+I, +R]]`` required by the flow-matching symbol update.
    These representations are not interchangeable.
    """
    H_nonherm = torch.transpose(torch.conj(H_complex), 2, 1).to(device=device)
    real_buffer[:, 0:NR, 0:NT] = torch.real(H_nonherm)
    real_buffer[:, 0:NR, NT:] = torch.imag(H_nonherm)
    real_buffer[:, NR:, 0:NT] = torch.imag(H_nonherm)
    real_buffer[:, NR:, NT:] = torch.real(H_nonherm)
    real_buffer[:, NR:, 0:NT] = -real_buffer[:, NR:, 0:NT]
