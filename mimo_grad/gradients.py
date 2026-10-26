"""Prior and measurement gradients for joint MIMO inference."""

from __future__ import annotations

import torch

from utils.util import gaussian


def get_channel_real_representation(
    H_complex: torch.Tensor, NR: int, NT: int
) -> torch.Tensor:
    """Return the real-valued block representation of a complex channel matrix.

    Real-imag block layout (Velocity / Flow-Matching path):
        [[ +real(H^H), -imag(H^H) ],
         [ +imag(H^H), +real(H^H) ]]
    where ``H^H = transpose(conj(H_complex))``.

    This is the representation used by the flow-matching symbol update.
    ``preparation.fill_real_channel_representation`` uses a different sign
    convention; its in-place result is not interchangeable with this one.
    """
    H_hermitian = torch.transpose(torch.conj(H_complex), 2, 1)
    H_real_repr = torch.zeros(
        (H_complex.shape[0], 2 * NR, 2 * NT), device=H_complex.device
    )
    real_part = torch.real(H_hermitian)
    imag_part = torch.imag(H_hermitian)
    H_real_repr[:, 0:NR, 0:NT] = real_part
    H_real_repr[:, 0:NR, NT:] = -imag_part
    H_real_repr[:, NR:, 0:NT] = imag_part
    H_real_repr[:, NR:, NT:] = real_part
    return H_real_repr

def compute_x_prior_gradient(
    x_current: torch.Tensor,
    generator,
    current_sigma_x: float,
    NT: int,
    M: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the prior score term for the symbol estimates.

    Evaluate the posterior-mean denoiser and subtract the current symbol state.
    """
    # x_current has shape (num_channels, 2 * NT, batch_size_x)
    num_channels, _, batch_size_x = x_current.shape

    x_gaussian = torch.transpose(x_current, 2, 1)
    Zi_hat = gaussian(
        x_gaussian.reshape(batch_size_x * num_channels, 2 * NT),
        generator,
        current_sigma_x**2,
        NT,
        M,
        device,
    )
    Zi_hat = torch.transpose(
        torch.reshape(Zi_hat, (num_channels, batch_size_x, 2 * NT)),
        2,
        1,
    )
    prior = (Zi_hat - x_current) / current_sigma_x**2
    return prior


def compute_x_likelihood_gradient(
    H_current_x: torch.Tensor,
    x_current: torch.Tensor,
    y_x: torch.Tensor,
    current_sigma_x: float,
    local_noise: float,
    identity: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the likelihood score term for the symbol estimates.

    Solve the symmetric positive-definite measurement covariance with a
    Cholesky factorization, then apply the transposed channel matrix.
    """
    H_dev = H_current_x.to(device=device)
    diff = (y_x - torch.matmul(H_dev, x_current)).float()
    cov_matrix = (
        (current_sigma_x**2)
        * torch.bmm(H_dev, torch.transpose(H_dev, 2, 1))
        + local_noise * identity
    )
    # Factor the symmetric positive-definite covariance.
    # torch.cholesky_solve(B, L) solves L @ L^T @ X = B, equivalent to C^{-1} @ B.
    # diff shape: [batch, NR, batch_x], matching cholesky_solve.
    L = torch.linalg.cholesky(cov_matrix.to(device=device))
    solved = torch.cholesky_solve(diff, L)
    grad_likelihood = torch.matmul(
        torch.transpose(H_dev, 2, 1),
        solved,
    )
    return grad_likelihood


def compute_h_likelihood_gradient(
    H_current: torch.Tensor,
    pilots: torch.Tensor,
    pilots_conj: torch.Tensor,
    x_current: torch.Tensor,
    y_pilots: torch.Tensor,
    y_H: torch.Tensor,
    sample_joint: bool,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the measurement gradient term for the channel estimates."""
    if sample_joint:
        forward_complex = x_current.chunk(2, dim=1)
        forward_complex = forward_complex[0] - 1j * forward_complex[1]
        forward_complex = torch.transpose(forward_complex, -1, -2)
        forward_H = torch.cat(
            (pilots_conj.to(device=device), forward_complex.to(device=device)),
            dim=1,
        )
        forward_herm = torch.conj(torch.transpose(forward_H, -1, -2)).to(device=device)
        meas_grad = torch.matmul(
            forward_herm,
            torch.matmul(forward_H, H_current.to(device=device)) - y_H,
        )
    else:
        meas_grad = torch.matmul(
            pilots,
            torch.matmul(pilots_conj, H_current.to(device=device)) - y_pilots,
        )
    return meas_grad


def compute_h_likelihood_gradient_split(
    H_current: torch.Tensor,
    pilots: torch.Tensor,
    pilots_conj: torch.Tensor,
    x_current: torch.Tensor,
    y_pilots: torch.Tensor,
    y_H: torch.Tensor,
    sample_joint: bool,
    device: torch.device | str,
    pilot_length: int = 30,
    data_length: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return measurement gradients separated into pilot and data contributions."""
    if sample_joint:
        forward_complex = x_current.chunk(2, dim=1)
        forward_complex = forward_complex[0] - 1j * forward_complex[1]
        forward_complex = torch.transpose(forward_complex, -1, -2)
        forward_H = torch.cat(
            (pilots_conj.to(device=device), forward_complex.to(device=device)),
            dim=1,
        )
        forward_herm = torch.conj(torch.transpose(forward_H, -1, -2)).to(device=device)
        residual = torch.matmul(forward_H, H_current.to(device=device)) - y_H

        expected_pilots = pilots_conj.shape[1]
        expected_data = forward_H.shape[1] - expected_pilots
        if pilot_length != expected_pilots or data_length != expected_data:
            raise ValueError(
                f"Requested pilot/data lengths ({pilot_length}, {data_length}) do not "
                f"match the concatenated matrix dimensions ({expected_pilots}, {expected_data})."
            )

        residual_pilot = residual[:, :pilot_length, :]
        residual_data = residual[:, pilot_length : pilot_length + data_length, :]

        forward_pilot = forward_H[:, :pilot_length, :]
        forward_data = forward_H[:, pilot_length : pilot_length + data_length, :]

        meas_grad_pilot = torch.matmul(
            torch.conj(torch.transpose(forward_pilot, -1, -2)), residual_pilot
        )
        meas_grad_data = torch.matmul(
            torch.conj(torch.transpose(forward_data, -1, -2)), residual_data
        )
        meas_grad_total = torch.matmul(forward_herm, residual)
    else:
        meas_grad_total = torch.matmul(
            pilots,
            torch.matmul(pilots_conj, H_current.to(device=device)) - y_pilots,
        )
        meas_grad_pilot = meas_grad_total
        meas_grad_data = torch.zeros_like(meas_grad_total)

    return meas_grad_total, meas_grad_pilot, meas_grad_data

def denoiser_fn(x_in, generator, sigma, NT, M, device):
    # Evaluate the posterior mean used by compute_x_prior_gradient.
    num_channels, _, batch_size_x = x_in.shape
    x_gaussian = torch.transpose(x_in, 2, 1)
    Zi_hat = gaussian(
        x_gaussian.reshape(batch_size_x * num_channels, 2 * NT),
        generator,
        sigma**2,
        NT,
        M,
        device,
    )
    Zi_hat = torch.transpose(
        torch.reshape(Zi_hat, (num_channels, batch_size_x, 2 * NT)),
        2,
        1,
    )
    # Return the denoised estimate before converting it to a score.
    return Zi_hat

def diffuser_fn(H_current_complex, diffuser_model, sigma_discrete, labels):
    """Denoise the complex channel with the pretrained diffusion score model."""
    # Convert complex channels to the model input layout [B, 2, H, W].
    H_current_real = torch.view_as_real(H_current_complex).permute(0, 3, 1, 2).contiguous()
    
    # The model returns a real-valued two-channel score.
    score_real = diffuser_model(H_current_real, labels)
    
    # Restore the complex layout by reversing the input transformation.
    score_complex = torch.view_as_complex(score_real.permute(0, 2, 3, 1).contiguous())
    
    # Apply the Tweedie posterior-mean estimate.
    H1_hat = H_current_complex + sigma_discrete**2 * score_complex
    
    return H1_hat
