import numpy as np
import torch
import sys
import os
import yaml
from dotmap import DotMap
import scipy.io as sio

sys.path.append("./")
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from utils.util import get_logger
from data.sample_generator import sample_generator


def load_and_setup_data(args):
    """
    Handles all the initial setup for the velocity-based experiment.
    Returns a dictionary containing all configured objects and parameters.
    """

    logger = get_logger()

    torch.cuda.empty_cache()
    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logger.info(f"Device set to {device}.")

    with open(args.test_config, "r") as f:
        test_config = DotMap(yaml.safe_load(f))
    logger.info(f"Loading training configuration from: {args.train_config}")
    with open(args.train_config, "r") as f:
        train_config = DotMap(yaml.safe_load(f))

    train_end_idx = train_config.data.train_samples + train_config.data.val_samples
    test_start_idx = test_config.test_data_start_index
    if test_start_idx < train_end_idx:
        logger.warning(
            "Warning: Test set (starts at %d) overlaps with training/validation set (ends at %d).",
            test_start_idx,
            train_end_idx,
        )

    test_seed = 4321
    result_dir = test_config.results_dir_velocity % test_seed
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    logger.info(f"Results will be saved to {result_dir}.")

    model_log_path = train_config.training.log_path % train_config.model.sigma_begin
    target_weights = os.path.join(model_log_path, train_config.training.model_name)
    logger.info(f"Loading trained model from: {target_weights}")
    contents = torch.load(target_weights, weights_only=False)

    config = contents["config"]
    config.sampling.steps_each = test_config.sampling_steps
    config.data.channel = args.channel
    config.model.step_size = 1 * 1e-10
    config.data.mod_n = test_config.modulation_order

    diffuser = NCSNv2Deepest(config)
    diffuser = diffuser.cuda()
    diffuser.load_state_dict(contents["model_state"])
    diffuser.eval()

    # Read the SNR interval as [start, stop, step] from the test configuration.
    snr_params = test_config.snr_range
    snr_range = np.arange(snr_params[0], snr_params[1], snr_params[2])
    noise_range = 10 ** (-snr_range / 10.0)

    NR = config.data.image_size[0]
    NT = config.data.image_size[1]
    M = int(np.sqrt(config.data.mod_n))

    num_channels = test_config.num_test_channels
    noise_boost = args.noise_boost
    logger.info(f"Size of the channels: {NR}x{NT}.")

    logger.info(f"Loading data from: {train_config.data.data_path}")
    if ".mat" in train_config.data.data_path:
        mat_contents = sio.loadmat(train_config.data.data_path)
        H = mat_contents["H_bank"]
    elif ".npz" in train_config.data.data_path:
        data_file = np.load(train_config.data.data_path)
        H = data_file["H_bank"]
    else:
        raise ValueError(
            f"Unsupported data file format in {train_config.data.data_path}"
        )

    test_end = test_start_idx + num_channels
    H_test_complex = torch.tensor(H[test_start_idx:test_end, :, :]).detach().numpy()

    generator = sample_generator(num_channels, config.data.mod_n, NR)

    aux = torch.tensor(H_test_complex)
    H_test_real_repr = torch.empty([num_channels, 2 * NR, 2 * NT])
    H_test_real_repr[:, 0:NR, 0:NT] = torch.real(aux)
    H_test_real_repr[:, 0:NR, NT:] = torch.imag(aux)
    H_test_real_repr[:, NR:, 0:NT] = torch.imag(aux)
    H_test_real_repr[:, NR:, NT:] = torch.real(aux)
    H_test_real_repr[:, :NR, NT:] = -H_test_real_repr[:, :NR, NT:]

    logger.info("Channels loaded.")

    return {
        "logger": logger,
        "device": device,
        "test_config": test_config,
        "train_config": train_config,
        "result_dir": result_dir,
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
