"""Utility functions for Flow-DLE."""

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from PIL import Image

from instaflow.pipeline_edit import InferenceState, RectifiedFlowStateMachine

from .config import DragConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flow_dle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 0xdeadbeef):
    """Set random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def np_to_latent(
    pipe: RectifiedFlowStateMachine,
    x: np.ndarray
) -> torch.Tensor:
    """Convert RGB numpy image to VAE latent space representation.
    
    Normalizes pixel values to [-1, 1], converts to tensor, and encodes 
    through the VAE encoder. Applies VAE scale factor for proper latent 
    normalization.
    
    Args:
        pipe: RectifiedFlowStateMachine pipeline instance
        x: RGB image as numpy array, shape [H, W, C], values in [0, 255]
    
    Returns:
        Latent tensor of shape [1, C_latent, H_latent, W_latent]
    
    Note:
        Uses torch.no_grad() to avoid computing gradients during encoding.
    """
    # Normalize to [-1, 1] range expected by VAE
    x = x.astype(np.float32) / 255.0
    x = (x * 2.0 - 1.0).transpose(2, 0, 1)  # HWC -> CHW
    
    # Convert to tensor and move to pipeline device/dtype
    x = torch.from_numpy(x).unsqueeze(0).to(device=pipe.device, dtype=pipe.dtype)
    
    # Encode through VAE and apply scale factor
    lat = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
    return lat


def invert_state_from_image(
    pipe: RectifiedFlowStateMachine,
    source_image: np.ndarray,
    prompt: str,
    num_inference_steps: int,
) -> InferenceState:
    """Create InferenceState from existing image via flow inversion.
    
    Encodes the source image into latent space, then runs rectified flow 
    inversion to compute the initial noise latent that would generate this 
    image. This enables editing of non-generated images (e.g., benchmark 
    samples) by treating them as if they were generated outputs.
    
    Args:
        pipe: RectifiedFlowStateMachine pipeline instance
        source_image: Input image as numpy array [H, W, C], values [0, 255]
        prompt: Text prompt for conditioning during inversion
        num_inference_steps: Number of inversion steps (higher = more accurate)
    
    Returns:
        InferenceState with inverted initial_latent ready for drag editing
    
    Note:
        The inverted state starts at step 0 with initial_latent set. 
        run_rf_drag will then run inference until drag_step before applying edits.
    """
    # Prepare state with image dimensions and prompt
    state = pipe.prepare_state(
        prompt=prompt,
        height=source_image.shape[0],
        width=source_image.shape[1],
        num_inference_steps=num_inference_steps,
    )
    
    # Encode source image to latent space
    state.latent = np_to_latent(pipe, source_image)
    state.i = num_inference_steps  # Start inversion from final step
    
    # Run flow inversion to recover initial noise latent
    state = pipe.invert_from_state(state)
    
    # Store inverted latent as initial condition for subsequent editing
    state.initial_latent = state.latent.detach().clone()
    
    return state


def pad_to_multiple(array: np.ndarray, divisor: int = 32, pad_value = 0):
    """
    Pads a numpy array to make height and width multiples of a given divisor.
    Padding is added to the bottom and right with a specified value.
    
    Args:
        array: numpy array representing an image (H x W) or (H x W x C)
        divisor: The number that dimensions should be multiples of (default: 32)
        pad_value: The value to use for padding (default: 0 for black)
    
    Returns:
        Padded numpy array with dimensions divisible by divisor
    """
    # Get original dimensions
    height, width = array.shape[:2]
    
    # Calculate padding needed for each dimension
    pad_height = (divisor - (height % divisor)) % divisor
    pad_width = (divisor - (width % divisor)) % divisor
    
    # Create padding specification for np.pad
    if len(array.shape) == 3:
        # RGB or RGBA image
        pad_spec = ((0, pad_height), (0, pad_width), (0, 0))
    else:
        # Grayscale image
        pad_spec = ((0, pad_height), (0, pad_width))
    
    # Apply padding
    padded_array = np.pad(
        array,
        pad_width=pad_spec,
        mode='constant',
        constant_values=pad_value
    )
    
    return padded_array


def load_dragbench_sample(
    sample_path: str
) -> Tuple[np.ndarray, str, np.ndarray, list]:
    """
    Load a DragBench sample.
    
    Args:
        sample_path: Path to sample directory
    
    Returns:
        Tuple of (image, prompt, mask, points)
    """
    try:
        # Load image
        image_path = os.path.join(sample_path, 'original_image.png')
        source_image = Image.open(image_path)
        source_image = np.array(source_image)
        
        # Load metadata
        meta_path = os.path.join(sample_path, 'meta_data.pkl')
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        
        prompt = meta_data['prompt']
        mask = meta_data['mask']
        points = meta_data['points']

        # DEBUG
        assert mask.min() == 0 and mask.max() == 1

        height, width, _ = source_image.shape
        if mask.shape != (height, width):
            raise RuntimeError(
                f"Mismatched dimensions: mask {mask.shape} != source {(height, width)}"
            )
        
        # Enforce image dimensions to be multiple of 32 for compatibility with latent scaling
        divisor = 32
        if width % divisor != 0 or height % divisor != 0:
            logger.debug(f"Sample shape is {source_image.shape}, padding to multiple of {divisor}")
            source_image = pad_to_multiple(source_image, divisor, pad_value=0)
            mask = pad_to_multiple(mask, divisor, pad_value=0)

        logger.debug(f"Loaded sample: {source_image.shape}, {len(points)} points")
        logger.debug(f"Mask shape {mask.shape}, mask sum: {mask.sum()}")
        logger.debug(f"Prompt: {prompt}")

        return source_image, prompt, mask, points
        
    except Exception as e:
        logger.error(f"Failed to load sample {sample_path}: {e}")
        raise


def save_results(
    output_dir: str,
    category: str,
    sample_name: str,
    drag_output,
    save_intermediates: bool = False
):
    """Save drag editing results."""
    save_dir = Path(output_dir) / category / sample_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final image
    final_img_path = save_dir / 'dragged_image.png'
    drag_output.final_image.save(final_img_path)
    logger.info(f"Saved final image: {final_img_path}")
    
    # Save intermediate visualizations if requested
    if save_intermediates and hasattr(drag_output, 'optim_steps'):
        for i, step_data in enumerate(drag_output.optim_steps):
            if hasattr(step_data, 'latent'):
                img = drag_output.pipe.decode_latents(
                    step_data.latent,
                    disable_safety_checker=True
                )[0][0]
                img.save(save_dir / f'intermediate_{i:03d}.png')
    
    # Save metadata
    metadata = {
        'drag_step': drag_output.drag_step,
        'end_step': drag_output.end_step,
        'converged': drag_output.converged,
        'total_steps': drag_output.total_steps,
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)


def setup_result_directory(
    result_dir: Optional[str],
    config: DragConfig
) -> Path:
    """Create result directory with descriptive name from DragConfig.
    
    Args:
        result_dir: Optional custom directory path
        config: DragConfig instance with all hyperparameters
    
    Returns:
        Path object for the result directory
    """
    if result_dir:
        result_path = Path(result_dir)
    else:
        # Format unet_feature_idx as hyphen-separated values (e.g., "3" or "3-5-7")
        unet_str = "-".join(str(idx) for idx in config.unet_feature_idx)
        
        # Format floats with consistent precision, avoiding dots in filenames
        lr_str = f"{config.lr:.2f}".replace(".", "p")  # e.g., "0p01"
        lam_str = f"{config.lam:.2f}".replace(".", "p")  # e.g., "0p50"
        
        result_name = (
            f"flow_dle_res_"
            f"drag{config.drag_step}_"
            f"npix{config.n_pix_step}_"
            f"lr{lr_str}_"
            f"lam{lam_str}_"
            f"unet{unet_str}"
        )
        result_path = Path(result_name)
    
    result_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Result directory: {result_path.absolute()}")
    return result_path