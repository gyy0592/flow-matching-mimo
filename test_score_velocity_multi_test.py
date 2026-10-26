import numpy as np
import torch
import sys
import os
import copy
import argparse

from pathlib import Path
from matplotlib import pyplot as plt
from rich.progress import Progress
from rich.console import Console
import scipy.io as sio
sys.path.append('./')

from data.loaders          import Channels
from torch.utils.data import DataLoader
from utils.util import *
from data.sample_generator import *
from mimo_grad.velocity_setup import load_and_setup_data
from mimo_grad.velocity_logging import IterationLogger
from mimo_grad import (
    compute_x_prior_gradient,
    compute_x_likelihood_gradient,
    compute_h_likelihood_gradient,
    compute_h_likelihood_gradient_split,
    select_snr_parameters,
    initialize_pilot_state,
    initialize_symbol_state,
    get_channel_real_representation,
    fill_real_channel_representation,
    denoiser_fn,
    diffuser_fn
)
from utils.seeding import set_seed_for_step, str2bool

# --- DEBUG CONTROLS ---
DEBUG_MODE = True  # Master switch for debug printing
LOG_FIRST_M_ITERS = 5  # Number of first iterations to log before NaN
LOG_LAST_N_ITERS = 5   # Number of last iterations to log before NaN
# --- END DEBUG CONTROLS ---

# Create a Rich Console that forces terminal output, bypassing detection.
console = Console(force_terminal=True)


def parse_args():
    """Parse command-line arguments for the velocity-flow MIMO test driver.

    Read optional defaults from configs/runtime.yaml. Explicit command-line
    arguments, including --seed, take precedence.
    """
    from utils.runtime_config import load_runtime_config
    _rc = load_runtime_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--channel', type=str, default='3GPP')
    parser.add_argument('--save_channels', type=int, default=0)
    parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[32 / 64])
    parser.add_argument('--noise_boost', nargs='+', type=float, default=0.001)
    parser.add_argument('--batch_size_x_list', nargs='+', type=float,
                        default=_rc.get('batch_size_x_list', [50]))
    parser.add_argument('--pilots_list', nargs='+', type=float,
                        default=_rc.get('pilots_list', [30]))
    parser.add_argument('--num_realizations', type=int,
                        default=_rc.get('num_realizations', 5),
                        help='Number of realizations')
    parser.add_argument('--sample_joint', type=str2bool, default=True,
                        help='(true/false or 1/0)')
    parser.add_argument('--train_config', type=str, required=True,
                        help='Path to the training config file for the model being tested')
    parser.add_argument('--test_config', type=str,
                        default='configs/testing_config_velocity.yaml',
                        help='Path to the testing config file')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Override sampling_params.num_steps from the test config')
    parser.add_argument('--realization_idx', type=int, default=None,
                        help='Run only specific realization (0-indexed), for parallelization')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting global random seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    num_realizations = max(1, int(args.num_realizations))
    args.num_realizations = num_realizations
    return args


def load_data_and_models(args):
    """Load test config, channel bank, and diffuser model. Returns a dict
    with all state needed by the SNR sweep loop. Preserves the original
    sequential order: setup, step-count override, then optional MAT override.
    """
    # --- Load data and setup ---
    setup_data = load_and_setup_data(args)
    logger = setup_data["logger"]
    device = setup_data["device"]
    test_config = setup_data["test_config"]
    train_config = setup_data["train_config"]
    base_result_dir = setup_data["result_dir"]
    config = setup_data["config"]
    diffuser = setup_data["diffuser"]

    # Use eager execution to preserve the numerical behavior of the score model.

    snr_range = setup_data["snr_range"]
    noise_range = setup_data["noise_range"]
    NR = setup_data["NR"]
    NT = setup_data["NT"]
    M = setup_data["M"]
    num_channels = setup_data["num_channels"]
    noise_boost = setup_data["noise_boost"]
    H_test_complex = setup_data["H_test_complex"]
    generator = setup_data["generator"]
    H_test_real_repr = setup_data["H_test_real_repr"]
    test_start_idx = setup_data["test_start_idx"]
    test_seed = setup_data["test_seed"]

    if args.num_steps is not None:
        logger.info(
            f"Overriding sampling_params.num_steps with command line value: {args.num_steps}"
        )
        test_config.sampling_params.num_steps = int(args.num_steps)

    # Optionally override the test channels with the original .mat bank.
    custom_mat_path = Path("data/H_bank_64_original.mat")
    if custom_mat_path.exists():
        logger.info(f"Overriding test channels with {custom_mat_path}.")
        mat_contents = sio.loadmat(str(custom_mat_path))
        H_full = mat_contents["H_bank"]
        test_end_idx = test_start_idx + num_channels
        H_test_complex = (
            torch.tensor(H_full[test_start_idx:test_end_idx, :, :]).detach().numpy()
        )
        aux = torch.tensor(H_test_complex)
        H_test_real_repr = torch.empty(
            (num_channels, 2 * NR, 2 * NT), dtype=aux.dtype
        )
        H_test_real_repr[:, 0:NR, 0:NT] = torch.real(aux)
        H_test_real_repr[:, 0:NR, NT:] = torch.imag(aux)
        H_test_real_repr[:, NR:, 0:NT] = torch.imag(aux)
        H_test_real_repr[:, NR:, NT:] = torch.real(aux)
        H_test_real_repr[:, :NR, NT:] = -H_test_real_repr[:, :NR, NT:]

    return {
        "logger": logger,
        "device": device,
        "test_config": test_config,
        "train_config": train_config,
        "base_result_dir": base_result_dir,
        "config": config,
        "diffuser": diffuser,
        "snr_range": snr_range,
        "noise_range": noise_range,
        "NR": NR,
        "NT": NT,
        "M": M,
        "num_channels": num_channels,
        "noise_boost": noise_boost,
        "H_test_complex": H_test_complex,
        "generator": generator,
        "H_test_real_repr": H_test_real_repr,
        "test_start_idx": test_start_idx,
        "test_seed": test_seed,
    }


def save_results(args, result_dir, batch_size_x_int, snr_range,
                 oracle_log, x_oracle_log, oracle_log_runs,
                 x_nmse_log_runs, SER_velocity_runs, logger):
    """Persist per-pilots-config sweep results to disk. Filename layout
    matches the parallel/sequential mode convention used by downstream
    aggregators."""
    if args.realization_idx is not None:
        save_filename = f"realization_{args.realization_idx}_batch{batch_size_x_int}.npz"
    else:
        save_filename = f"all_realizations_batch{batch_size_x_int}.npz"

    save_path = os.path.join(result_dir, save_filename)
    np.savez(
        save_path,
        snr_range=snr_range,
        oracle_log=oracle_log,
        x_oracle_log=x_oracle_log,
        oracle_log_runs=oracle_log_runs,
        x_nmse_log_runs=x_nmse_log_runs,
        SER_velocity_runs=SER_velocity_runs,
    )
    logger.info(f"Results saved to {save_path}")


def run_snr_sweep(args, data):
    """Drive the symbol-count, pilot-count, SNR and realization loops.
    Calls save_results() once per (batch_size_x, pilots) configuration."""
    logger = data["logger"]
    device = data["device"]
    test_config = data["test_config"]
    base_result_dir = data["base_result_dir"]
    config = data["config"]
    diffuser = data["diffuser"]
    snr_range = data["snr_range"]
    noise_range = data["noise_range"]
    NR = data["NR"]
    NT = data["NT"]
    M = data["M"]
    num_channels = data["num_channels"]
    H_test_complex = data["H_test_complex"]
    generator = data["generator"]
    H_test_real_repr = data["H_test_real_repr"]
    test_seed = data["test_seed"]

    num_realizations = args.num_realizations

    # Main loop -- inference
    for batch_size_x in args.batch_size_x_list:
        batch_size_x_int = int(float(batch_size_x))
        logger.info(f"Starting experiment with this number of symbols: {batch_size_x_int}.")
        for pilots in args.pilots_list:
            pilots_int = int(float(pilots))
            logger.info(f"Starting experiment with this number of pilots: {pilots_int}.")

            SER_velocity = []
            H_list = []
            # Set some hyperparameters
            config.data.num_pilots = pilots_int
            print(config.data.num_pilots)

            # Load data
            dataset_pilots = Channels(test_seed, config, H=H_test_complex, norm="global")
            batch_size = len(dataset_pilots)
            loader = DataLoader(dataset_pilots, batch_size=num_channels,
                                shuffle=False, num_workers=0, drop_last=True)

            iter_ = iter(loader)
            samples_pilots = next(iter_)
            _, pilots, _ = (
                samples_pilots['H'].cuda(),
                samples_pilots['P'].cuda(),
                samples_pilots['Y'].cuda(),
            )

            pilots_conj = torch.conj(torch.transpose(pilots, -1, -2))
            H_herm = samples_pilots['H_herm'].cuda()
            H_herm_complex = H_herm[:, 0] + 1j * H_herm[:, 1]
            # t / (1-t) = snr -> t = snr / (1+snr)

            # Preallocate metric logs across the SNR sweep to avoid overwriting earlier rows.
            ode_steps = test_config.sampling_params.num_steps
            result_dir = os.path.join(base_result_dir, f"steps{ode_steps}")
            os.makedirs(result_dir, exist_ok=True)
            total_iter = max(1, ode_steps)
            oracle_log = np.zeros((len(snr_range), total_iter))
            x_oracle_log = np.zeros((len(snr_range), total_iter))

            # Store one run for a selected realization, otherwise all requested runs.
            actual_num_runs = 1 if args.realization_idx is not None else num_realizations
            oracle_log_runs = np.zeros((actual_num_runs, len(snr_range), total_iter))
            x_nmse_log_runs = np.zeros_like(oracle_log_runs)
            SER_velocity_runs = np.zeros((actual_num_runs, len(snr_range)))
            H_estimated_runs = [[None for _ in range(len(snr_range))] for _ in range(actual_num_runs)]

            # Sweep the configured SNR values.

            # Start the loop for all SNRs
            for snr_idx, local_noise in enumerate(noise_range):
                # 1. Reset seed for SNR Step
                # Use fixed seed for Problem Instance (H, X, Y) generation across all SNRs
                # This ensures we test the Exact Same Channel/Data instance at 10dB and 20dB (Common Random Numbers)
                # which produces smoother curves and ensures consistency regardless of SNR range.
                # We still mix in batch_size to differ between batch experiments (though H is same, X differs).
                seed_snr = set_seed_for_step(args.seed, batch_size_x_int)

                # Construct the two-stage OT time schedule.
                t_start = 0.2
                t_end = 0.9999  # End time is close to 1 to avoid singularity
                t1ratio = 0.35
                num_step1 = int(ode_steps * t1ratio)
                num_step2 = ode_steps - num_step1
                t_schedule1 = np.linspace(t_start, t_end, num_step1)
                t_schedule2 = np.linspace(t_start, t_end, num_step2)
                # t_schedule2 = np.linspace(0.9999, 0.99999, num_step2 )
                # concatenate the two schedules
                t_schedule = np.concatenate((t_schedule1, t_schedule2))
                dts1_intermediate = np.diff(t_schedule1)
                # Repeat the preceding interval for the last point.
                last_dt1 = dts1_intermediate[-1] if len(dts1_intermediate) > 0 else 0.0
                # Match the interval array to the first schedule length.
                dts1 = np.append(dts1_intermediate, last_dt1)

                # Compute intervals for the second schedule in the same way.
                dts2_intermediate = np.diff(t_schedule2)
                last_dt2 = dts2_intermediate[-1] if len(dts2_intermediate) > 0 else 0.0
                dts2 = np.append(dts2_intermediate, last_dt2)

                # Concatenate the interval arrays.
                dts = np.concatenate((dts1, dts2))
                logger.info(
                    f"Setting up ODE solver for OT path from t={t_start} to t={t_end} in {ode_steps} steps."
                )

                all_sigmas_ve = diffuser.sigmas.cpu().numpy()
                all_sigmas_ve[all_sigmas_ve == 0] = 1e-10
                all_snr_ve = 1.0 / all_sigmas_ve

                snr_ot_schedule = t_schedule / (1.0 - t_schedule)
                sigma_diffusion_t_vector = np.interp(snr_ot_schedule, all_snr_ve, all_sigmas_ve)

                logger.info(f"OT schedule t[:5]: {t_schedule[:5]}")
                logger.info(f"Corresponding sigma_t'[:5]: {sigma_diffusion_t_vector[:5]}")

                # Configure the symbol and pilot measurement updates.

                # Setting parameters for each SNR
                Id = batch_identity_matrix(2 * NR, 2 * NR, batch_size).to(device=device)
                temp_x, sigmas_x, epsilon = select_snr_parameters(
                    snr_range[snr_idx], test_config, len(t_schedule)
                )

                # Prepare data associated to the pilots
                y_pilots, H_current, oracle = initialize_pilot_state(
                    pilots_conj, H_herm_complex, local_noise
                )

                # Prepare data associated to the symbols
                (
                    x_current,
                    j_indices,
                    y_x,
                    H_current_x,
                    y_H,
                ) = initialize_symbol_state(
                    H_test_real_repr,
                    generator,
                    num_channels,
                    NR,
                    NT,
                    batch_size_x_int,
                    local_noise,
                    device,
                    y_pilots,
                )
                initial_sigma_ot = 1.0 - t_start
                H_current = H_current * initial_sigma_ot
                H_initial = H_current.clone()
                x_init_list = [torch.randn_like(x_current) for _ in range(num_realizations)]
                oracle = H_herm_complex
                H_sum = None

                # Optionally execute one selected realization.
                if args.realization_idx is not None:
                    # Execute only the selected realization.
                    realization_indices = [args.realization_idx]
                    if args.realization_idx >= num_realizations:
                        raise ValueError(
                            f"realization_idx {args.realization_idx} >= num_realizations {num_realizations}"
                        )
                else:
                    # Execute all requested realizations.
                    realization_indices = range(num_realizations)

                for realization_idx in realization_indices:
                    # 2. Reset seed for Realization Step (Solver Consistency)
                    # Use simple arithmetic as requested: Base + Offset + Realization
                    # Offset ensures solver noise is distinct from Channel generation noise (defined by seed_snr)
                    # Realization_idx ensures each realization is independent.
                    set_seed_for_step(args.seed + 1000000 + realization_idx, batch_size_x_int)

                    iteration_logger = IterationLogger(DEBUG_MODE, LOG_FIRST_M_ITERS, LOG_LAST_N_ITERS)
                    nan_detected_and_printed = False
                    H_current = H_initial.clone()
                    x_current = x_init_list[realization_idx].clone()
                    iter_lang = 0
                    oracle_row = np.zeros(total_iter)
                    x_nmse_row = np.zeros(total_iter)

                    with torch.no_grad():
                        with Progress(console=console) as progress:
                            ode_task = progress.add_task(
                                f"[green]SNR {snr_range[snr_idx]:.1f}dB (run {realization_idx + 1}/{num_realizations})...",
                                total=ode_steps,
                            )

                            for i in range(len(t_schedule)):
                                t_current = t_schedule[i]
                                dt = dts[i]
                                # --- Unconditional Velocity Calculation ---

                                # Match the channel state to the VE noise schedule.
                                sigma_interpolated = sigma_diffusion_t_vector[i]

                                # --- X Schedule ---
                                current_sigma_x = sigmas_x[min(i, len(sigmas_x) - 1)]
                                step_x = epsilon * (current_sigma_x / sigmas_x[-1]) ** 2
                                # Select the nearest discrete score-model noise level.
                                label_idx = np.abs(all_sigmas_ve - sigma_interpolated).argmin()
                                sigma_discrete = all_sigmas_ve[label_idx]
                                labels = torch.ones(
                                    H_current.shape[0], device=device, dtype=torch.long
                                ) * label_idx

                                # Compute the time-dependent uncertainty from paper Eq. (13).
                                alpha_ot_t = t_current
                                sigma_ot_t = 1.0 - t_current
                                r_t_squared = (sigma_ot_t ** 2) / (sigma_ot_t ** 2 + alpha_ot_t ** 2 + 1e-9)

                                max_x_steps = 6
                                x_inner_steps = max_x_steps
                                step_x = step_x / 2

                                # The channel is fixed during the symbol inner loop;
                                # construct its real representation once per channel update.
                                H_current_x = get_channel_real_representation(
                                    H_current, NR, NT
                                ).to(device=device)
                                # snr_linear_10 is also loop-invariant; compute once.
                                snr_linear_10 = 10 ** (10 / 10)
                                _noise_floor_x = max(local_noise, 1.0 / snr_linear_10)

                                for inner_idx in range(x_inner_steps):

                                    grad_prior_x = compute_x_prior_gradient(
                                        x_current,
                                        generator,
                                        current_sigma_x,
                                        # sigma_ot_t,
                                        NT,
                                        M,
                                        device,
                                    )
                                    x_current = x_current + grad_prior_x * step_x

                                    grad_like_x = compute_x_likelihood_gradient(
                                        H_current_x,
                                        x_current,
                                        y_x,
                                        current_sigma_x,
                                        _noise_floor_x,  # pre-computed outside inner loop
                                        Id,
                                        device,
                                    )

                                    likelihood_correction_x = grad_like_x
                                    x_current = x_current + likelihood_correction_x * step_x
                                    noise_scale = np.sqrt(2 * temp_x * step_x)
                                    noise_x = torch.randn_like(x_current) * noise_scale
                                    x_current = x_current + noise_x

                                H0_hat = diffuser_fn(H_current / t_current, diffuser, sigma_discrete, labels)
                                v_uncond_H = (H0_hat - H_current) / (1.0 - t_current)
                                H_current = H_current + v_uncond_H * dt

                                pilot_length = int(pilots_conj.shape[1])
                                data_length = int(y_H.shape[1] - pilot_length) if args.sample_joint else 0
                                (
                                    _,
                                    grad_like_h_pilot,
                                    grad_like_h_data,
                                ) = compute_h_likelihood_gradient_split(
                                    H_current,
                                    pilots,
                                    pilots_conj,
                                    x_current=x_current,
                                    y_pilots=y_pilots,
                                    y_H=y_H,
                                    sample_joint=args.sample_joint,
                                    device=device,
                                    pilot_length=pilot_length,
                                    data_length=data_length,
                                )
                                grad_like_h = grad_like_h_pilot + grad_like_h_data * (
                                    t_current if i > ode_steps * t1ratio else 0.0
                                )

                                g_like_H = -grad_like_h / (local_noise / 2.0 + r_t_squared)
                                correction_term = ((1.0 - t_current) / (t_current + 1e-9)) * g_like_H
                                real_step = correction_term * dt
                                max_step_norm = 2

                                def _clip_step_per_sample(step, limit):
                                    step_to_clip = step
                                    if torch.is_complex(step_to_clip):
                                        step_to_clip = torch.view_as_real(step_to_clip)
                                    flat = step_to_clip.reshape(step_to_clip.shape[0], -1)
                                    norms = torch.linalg.norm(flat, dim=1)
                                    scale = torch.clamp(limit / (norms + 1e-9), max=1.0)
                                    return step * scale.view(-1, 1, 1), norms

                                real_step, step_norms = _clip_step_per_sample(
                                    real_step, max_step_norm
                                )
                                if args.sample_joint and DEBUG_MODE and i < LOG_FIRST_M_ITERS:
                                    console.print(
                                        f"[debug] real_step_norm={step_norms.max():.3e}"
                                    )
                                H_current = H_current + real_step

                                x_current_flat = torch.transpose(x_current, -1, -2).reshape(
                                    num_channels * batch_size_x_int, 2 * NT
                                )
                                j_indices_gpu = j_indices.to(device=x_current.device, dtype=torch.long)
                                constellation_gpu = generator.constellation.to(device=x_current.device)
                                real_idx = j_indices_gpu // M
                                imag_idx = j_indices_gpu % M
                                x_true_flat = torch.cat(
                                    [constellation_gpu[real_idx], constellation_gpu[imag_idx]], dim=-1
                                )
                                oracle_row[iter_lang] = torch.mean(
                                    (torch.sum(torch.square(torch.abs(H_current.to(device='cpu') - oracle.to(device='cpu'))), dim=(-1, -2)) /
                                     torch.sum(torch.square(torch.abs(oracle.to(device='cpu'))), dim=(-1, -2)))
                                ).cpu().numpy()
                                x_nmse_row[iter_lang] = (
                                    torch.mean(torch.sum((x_current_flat - x_true_flat) ** 2, dim=-1) /
                                               (torch.sum(x_true_flat ** 2, dim=-1) + 1e-9))
                                ).item()
                                if i % 10 == 0:
                                    ser_debug = 1.0 - sym_detection(
                                        x_current_flat.to(device="cpu"),
                                        j_indices.to(device="cpu"),
                                        generator.real_QAM_const,
                                        generator.imag_QAM_const,
                                    )
                                    nmse_H_debug = oracle_row[iter_lang]
                                    nmse_X_debug = x_nmse_row[iter_lang]
                                    console.print(
                                        f"[debug] step = {i}  NMSE_H={nmse_H_debug:.6e} NMSE_X={nmse_X_debug:.6e}  SER= {ser_debug:.6f} "
                                    )
                                iter_lang += 1
                                progress.update(ode_task, advance=1)

                    ser_value = 1 - sym_detection(
                        torch.transpose(x_current, -1, -2).reshape(num_channels * batch_size_x_int, 2 * NT).to(device='cpu'),
                        j_indices,
                        generator.real_QAM_const,
                        generator.imag_QAM_const,
                    )
                    if isinstance(ser_value, torch.Tensor):
                        ser_value = ser_value.item()

                    # A selected run occupies index zero; sequential runs retain their index.
                    save_idx = 0 if args.realization_idx is not None else realization_idx

                    SER_velocity_runs[save_idx, snr_idx] = ser_value
                    H_estimated_runs[save_idx][snr_idx] = H_current.detach().cpu()
                    if H_sum is None:
                        H_sum = H_current.detach().clone()
                    else:
                        H_sum = H_sum + H_current.detach()
                    oracle_log_runs[save_idx, snr_idx, :] = oracle_row
                    x_nmse_log_runs[save_idx, snr_idx, :] = x_nmse_row

                    final_error_for_snr = oracle_row[iter_lang - 1] if iter_lang > 0 else 0
                    final_error_db = 10 * np.log10(final_error_for_snr) if final_error_for_snr > 0 else -np.inf
                    final_nmse_x_db = (
                        10 * np.log10(x_nmse_row[iter_lang - 1])
                        if iter_lang > 0 and x_nmse_row[iter_lang - 1] > 0 else -np.inf
                    )
                    print(
                        f"SNR {snr_range[snr_idx]:.1f} dB run {realization_idx + 1}/{num_realizations}: "
                        f"Final NMSE_H = {final_error_db:.2f} dB, Final NMSE_X = {final_nmse_x_db:.2f} dB, SER = {ser_value:.6f}"
                    )

                H_list.append(H_sum / num_realizations)
                SER_velocity.append(float(np.mean(SER_velocity_runs[:, snr_idx])))
                # Save results
                avg_oracle_snr = np.mean(oracle_log_runs[:, snr_idx, :], axis=0)
                avg_x_nmse_snr = np.mean(x_nmse_log_runs[:, snr_idx, :], axis=0)
                oracle_log[snr_idx, :] = avg_oracle_snr
                x_oracle_log[snr_idx, :] = avg_x_nmse_snr

            save_results(
                args, result_dir, batch_size_x_int, snr_range,
                oracle_log, x_oracle_log, oracle_log_runs,
                x_nmse_log_runs, SER_velocity_runs, logger,
            )


def main():
    args = parse_args()
    data = load_data_and_models(args)
    run_snr_sweep(args, data)


if __name__ == "__main__":
    main()
