import argparse
import datetime
import logging
import math
import os
import random
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from learning_brain import LearningBrain
from org import Config, CognitiveOrganism, cpp_loader, init_state


def _require_config(name: str):
    if not hasattr(Config, name):
        raise RuntimeError(f"Missing required Config.{name} from C++ firmware.")
    return getattr(Config, name)


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "training_debug.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints"
DATA_DIR = "."
SEQ_LEN = int(_require_config("SEQ_LEN"))
BATCH_SIZE = int(_require_config("BATCH_SIZE"))
EPOCHS = int(_require_config("EPOCHS"))
SEED = int(_require_config("SEED"))
WORKING_DIM = int(_require_config("WORKING_DIM"))
L_DIM = int(_require_config("L"))
R_DIM = int(_require_config("R"))
C_DIM = int(_require_config("C"))
DEVICE = torch.device(str(_require_config("DEVICE")))

if DEVICE.type != "cpu":
    raise RuntimeError(
        "This training pipeline is CPU-only. "
        f"Resolved Config.DEVICE='{DEVICE}', expected 'cpu'."
    )


class ByteDataset(Dataset):
    _file_cache: Dict[str, bytes] = {}
    _byte_bits_lut: Optional[torch.Tensor] = None

    def __init__(
        self,
        file_path: str,
        seq_len: int,
        start_offset: int,
        end_offset: int,
        random_sampling: bool,
        samples_per_epoch: Optional[int] = None,
    ):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if file_path not in ByteDataset._file_cache:
            with open(file_path, "rb") as f:
                ByteDataset._file_cache[file_path] = f.read()
        self.data = ByteDataset._file_cache[file_path]

        self.seq_len = int(seq_len)
        self.total_bytes = len(self.data)
        self.start_offset = max(0, int(start_offset))
        self.end_offset = min(self.total_bytes, int(end_offset))
        self.random_sampling = bool(random_sampling)
        self.vocab_size = 256

        if self.end_offset <= self.start_offset:
            raise RuntimeError(
                "Invalid dataset span: "
                f"start_offset={self.start_offset}, end_offset={self.end_offset}."
            )

        self.window_count = self.end_offset - self.start_offset - self.seq_len
        if self.window_count <= 0:
            raise RuntimeError(
                "Dataset span too small for sequence sampling. "
                f"seq_len={self.seq_len}, span=[{self.start_offset}, {self.end_offset})."
            )

        default_samples = max(1, self.window_count // max(1, self.seq_len))
        if samples_per_epoch is None:
            self.sample_count = default_samples
        else:
            requested = int(samples_per_epoch)
            if requested <= 0:
                raise RuntimeError(f"samples_per_epoch must be > 0, got {requested}.")
            self.sample_count = min(requested, self.window_count)

        if ByteDataset._byte_bits_lut is None:
            values = torch.arange(256, dtype=torch.long)
            shifts = torch.arange(7, -1, -1, dtype=torch.long)
            ByteDataset._byte_bits_lut = ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
        self.byte_bits_lut = ByteDataset._byte_bits_lut

        logger.info(
            "ByteDataset loaded: file=%s total_bytes=%d range=[%d,%d) random=%s samples=%d",
            file_path,
            self.total_bytes,
            self.start_offset,
            self.end_offset,
            self.random_sampling,
            self.sample_count,
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_sampling:
            start = random.randint(self.start_offset, self.end_offset - self.seq_len - 1)
        else:
            stride = max(1, self.window_count // max(1, self.sample_count))
            start = self.start_offset + ((int(idx) * stride) % self.window_count)

        chunk = self.data[start : start + self.seq_len + 1]
        if len(chunk) != self.seq_len + 1:
            raise RuntimeError(
                "Invalid byte chunk length in ByteDataset.__getitem__. "
                f"start={start} expected={self.seq_len + 1} got={len(chunk)}"
            )

        byte_values = np.frombuffer(chunk, dtype=np.uint8).copy()
        bits = self.byte_bits_lut[torch.from_numpy(byte_values).long()]
        x_bits = bits[:-1].clone()
        y_bits = bits[1:].clone()
        return x_bits, y_bits


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    seq_len: int,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    file_path = os.path.join(data_dir, "dataset.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing required training file: {file_path}. "
            "Create dataset.txt before launching training."
        )

    total_bytes = os.path.getsize(file_path)
    if total_bytes <= seq_len + 1:
        raise RuntimeError(
            f"dataset.txt too small ({total_bytes} bytes) for seq_len={seq_len}."
        )

    min_non_overlap = 2 * (seq_len + 1)
    if total_bytes < min_non_overlap:
        raise RuntimeError(
            "dataset.txt is too small for strict non-overlapping train/val split. "
            f"bytes={total_bytes} required={min_non_overlap}"
        )

    split_ratio = float(val_split)
    if not (0.01 <= split_ratio <= 0.5):
        raise RuntimeError(
            f"val_split must be in [0.01, 0.5], got {split_ratio}."
        )

    split_byte = int(total_bytes * (1.0 - split_ratio))
    split_byte = max(seq_len + 1, min(total_bytes - (seq_len + 1), split_byte))

    train_samples_cfg = _require_config("TRAIN_SAMPLES_PER_EPOCH")
    val_samples_cfg = _require_config("VAL_SAMPLES_PER_EPOCH")

    train_default = max(1, (split_byte - seq_len) // max(1, seq_len))
    val_default = max(1, (total_bytes - split_byte - seq_len) // max(1, seq_len))

    train_ds = ByteDataset(
        file_path=file_path,
        seq_len=seq_len,
        start_offset=0,
        end_offset=split_byte,
        random_sampling=True,
        samples_per_epoch=(train_default if train_samples_cfg is None else int(train_samples_cfg)),
    )
    val_ds = ByteDataset(
        file_path=file_path,
        seq_len=seq_len,
        start_offset=split_byte,
        end_offset=total_bytes,
        random_sampling=False,
        samples_per_epoch=(val_default if val_samples_cfg is None else int(val_samples_cfg)),
    )

    cpu_workers = os.cpu_count() or 1
    if os.name == "nt":
        train_workers = 0
        val_workers = 0
    else:
        train_workers = min(4, max(0, cpu_workers - 1))
        val_workers = min(2, max(0, cpu_workers - 1))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=train_workers,
        pin_memory=False,
        persistent_workers=(train_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=False,
        persistent_workers=(val_workers > 0),
    )
    return train_loader, val_loader


def set_global_seed(seed: int) -> None:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"{name} is not a tensor. got={type(tensor).__name__}")
    if torch.isfinite(tensor).all():
        return
    bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
    sample = bad[:5].cpu().tolist()
    raise RuntimeError(
        f"Non-finite values detected in {name}. "
        f"shape={tuple(tensor.shape)} sample_bad_indices={sample}"
    )


def run_preflight_checks(model: CognitiveOrganism, learning_brain: LearningBrain, device: torch.device) -> None:
    if cpp_loader is None:
        raise RuntimeError("cpp_loader extension is required but unavailable.")

    required_ops = (
        "unified_dispatch_io",
        "mes_super_step_io",
        "survival_losses_io",
        "quantized_matmul",
    )
    missing = [op for op in required_ops if not hasattr(cpp_loader, op)]
    if missing:
        raise RuntimeError(
            "Missing required cpp_loader operations: " + ", ".join(missing)
        )

    if device.type != "cpu":
        raise RuntimeError(
            f"Preflight failed: device '{device}' is unsupported. Expected CPU tensors only."
        )

    model.eval()
    with torch.no_grad():
        xb = torch.randint(0, 2, (2, 16, 8), device=device, dtype=torch.float32)
        H = init_state(model.L, model.R, model.d_s2, model.C, device=device).unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
        out, H_next, cost, gate = model(xb, H)
        _assert_finite_tensor("preflight.out", out)
        _assert_finite_tensor("preflight.H_next", H_next)
        _assert_finite_tensor("preflight.cost", cost)
        _assert_finite_tensor("preflight.gate", gate)

        mes_info = model.mes_step(xb, xb, precomputed_H_next=H_next)
        if not isinstance(mes_info, dict) or "mes_loss" not in mes_info:
            raise RuntimeError(
                f"mes_step returned invalid payload: type={type(mes_info).__name__}"
            )
        mes_loss = float(mes_info["mes_loss"])
        if not math.isfinite(mes_loss):
            raise RuntimeError(f"mes_step produced non-finite mes_loss={mes_loss}.")

        stab, eng, coh = learning_brain.calculate_metabolic_losses(H_next, H_prev=H)
        _assert_finite_tensor("preflight.loss_stability", stab)
        _assert_finite_tensor("preflight.loss_energy", eng)
        _assert_finite_tensor("preflight.loss_coherence", coh)

    model.train()
    logger.info("Preflight checks passed (C++ ops + forward + MES + survival losses).")


class Trainer:
    def __init__(self, model: CognitiveOrganism, learning_brain: LearningBrain, device: torch.device):
        self.model = model
        self.learning_brain = learning_brain
        self.device = device

        self.global_step = 0
        self.current_epoch = 0
        self.start_batch_idx = 0
        self.total_time = 0.0
        self.session_start = time.time()

        logger.info(
            "Trainer initialized: MES-driven C++ training pipeline active."
        )

    def _new_state(self, batch_size: int) -> torch.Tensor:
        return init_state(
            self.model.L,
            self.model.R,
            self.model.d_s2,
            self.model.C,
            device=self.device,
        ).unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, int, bool]:
        if max_steps is not None and int(max_steps) <= 0:
            raise RuntimeError(f"max_steps must be > 0 when provided, got {max_steps}.")

        self.model.train()
        total_loss = 0.0
        processed_batches = 0
        steps_this_call = 0
        reached_step_limit = False
        last_batch_idx = self.start_batch_idx - 1
        log_freq = max(1, len(dataloader) // 5)

        for batch_idx, (xb, yb) in enumerate(dataloader):
            if batch_idx < self.start_batch_idx:
                continue
            if max_steps is not None and steps_this_call >= int(max_steps):
                reached_step_limit = True
                break

            xb = xb.to(self.device, dtype=torch.float32)
            yb = yb.to(self.device, dtype=torch.float32)
            H = self._new_state(xb.size(0))

            with torch.no_grad():
                out, H_next, cost_step, gate = self.model(xb, H)
                _assert_finite_tensor("train.out", out)
                _assert_finite_tensor("train.H_next", H_next)
                _assert_finite_tensor("train.cost_step", cost_step)
                _assert_finite_tensor("train.gate", gate)

                task_loss_t = self.learning_brain.calculate_task_loss(out, yb)
                _assert_finite_tensor("train.task_loss", task_loss_t)
                stab_t, eng_t, coh_t = self.learning_brain.calculate_metabolic_losses(H_next, H_prev=H)
                _assert_finite_tensor("train.loss_stability", stab_t)
                _assert_finite_tensor("train.loss_energy", eng_t)
                _assert_finite_tensor("train.loss_coherence", coh_t)

                mes_info = self.model.mes_step(xb, yb, precomputed_H_next=H_next)
                if not isinstance(mes_info, dict) or "mes_loss" not in mes_info:
                    raise RuntimeError(
                        "MES step returned invalid payload during training. "
                        f"type={type(mes_info).__name__} payload={mes_info}"
                    )
                mes_loss = float(mes_info["mes_loss"])
                if not math.isfinite(mes_loss):
                    raise RuntimeError(
                        "MES step returned non-finite mes_loss. "
                        f"epoch={epoch_idx} batch={batch_idx} global_step={self.global_step} mes_loss={mes_loss}"
                    )

                # Total loss is tracked for reporting; parameter updates are executed by MES in C++.
                task_loss = float(task_loss_t.item())
                stability_loss = float(stab_t.item())
                energy_loss = float(eng_t.item())
                step_cost = float(cost_step.mean().item()) if cost_step.numel() > 0 else 0.0
                survival_weight = float(_require_config("SURVIVAL_WEIGHT"))
                energy_weight = float(_require_config("DYNAMIC_ENERGY_SCALE"))
                step_loss = task_loss + (survival_weight * stability_loss) + (energy_weight * (energy_loss + step_cost))

            total_loss += step_loss
            processed_batches += 1
            steps_this_call += 1
            last_batch_idx = batch_idx
            self.global_step += 1

            if batch_idx % log_freq == 0:
                elapsed = self.total_time + (time.time() - self.session_start)
                elapsed_text = str(datetime.timedelta(seconds=int(elapsed)))
                logger.info(
                    "Epoch %d | Batch %d | Global Step %d | Omega %.4f | Loss %.6f | Time %s",
                    epoch_idx,
                    batch_idx,
                    self.global_step,
                    float(self.model.omega),
                    step_loss,
                    elapsed_text,
                )

        self.start_batch_idx = (last_batch_idx + 1) if reached_step_limit else 0
        if processed_batches == 0:
            raise RuntimeError(
                "No training batches were processed in train_epoch. "
                f"epoch={epoch_idx} start_batch_idx={self.start_batch_idx}"
            )

        return total_loss / processed_batches, steps_this_call, reached_step_limit

    def validate(self, dataloader: DataLoader, train_loss: float) -> float:
        self.model.eval()
        val_total = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.float32)
                H = self._new_state(xb.size(0))
                out, _, _, _ = self.model(xb, H)
                batch_loss = self.learning_brain.calculate_task_loss(out, yb)
                _assert_finite_tensor("validation.loss", batch_loss)
                val_total += float(batch_loss.item())
                val_count += 1

        self.model.train()
        if val_count == 0:
            raise RuntimeError("Validation dataloader yielded zero batches.")

        val_loss = val_total / val_count
        old_omega = float(self.model.omega)
        new_omega = float(self.model.update_omega(float(train_loss), float(val_loss)))
        self.model.governor.evolutionary_step(
            self.model,
            {
                "val_loss": float(val_loss),
                "tps_pressure": float(self.model._get_tps_pressure()),
            },
        )
        logger.info(
            "Validation complete: train_loss=%.6f val_loss=%.6f omega=%.4f->%.4f",
            float(train_loss),
            float(val_loss),
            old_omega,
            new_omega,
        )
        return val_loss

    def save_checkpoint(self, path: str, batch_idx: int) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "current_phase": int(self.model.current_phase),
            "omega": float(self.model.omega),
            "global_step": int(self.global_step),
            "epoch": int(self.current_epoch),
            "batch_idx": int(batch_idx),
            "total_time": float(self.total_time + (time.time() - self.session_start)),
            "rng_state": torch.get_rng_state(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        self.cleanup_old_checkpoints(os.path.dirname(path))

    def cleanup_old_checkpoints(self, checkpoint_dir: str) -> None:
        if not os.path.isdir(checkpoint_dir):
            return

        epoch_files = []
        for name in os.listdir(checkpoint_dir):
            if not name.endswith(".pt"):
                continue
            if name in {"best_model.pt", "bios_best.pt", "latest.pt"}:
                continue
            match = re.match(r"^(?:train|rra)_epoch_(\d+)\.pt$", name)
            if match is None:
                continue
            epoch_files.append((int(match.group(1)), name))

        epoch_files.sort(key=lambda item: item[0])
        for _, old_name in epoch_files[:-1]:
            old_path = os.path.join(checkpoint_dir, old_name)
            os.remove(old_path)
            logger.info("Deleted old checkpoint: %s", old_path)

    def load_checkpoint(self, path: str) -> bool:
        if not os.path.exists(path):
            return False

        ckpt = torch.load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise RuntimeError(
                f"Checkpoint payload has invalid type {type(ckpt).__name__}; expected dict."
            )

        required_keys = {
            "model_state_dict",
            "current_phase",
            "omega",
            "global_step",
            "epoch",
            "batch_idx",
            "total_time",
            "rng_state",
        }
        missing_required = [k for k in required_keys if k not in ckpt]
        if missing_required:
            raise RuntimeError(
                "Checkpoint missing required keys: " + ", ".join(missing_required)
            )

        state_dict = ckpt["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "Checkpoint key 'model_state_dict' must be dict, "
                f"got {type(state_dict).__name__}."
            )

        current_state = self.model.state_dict()
        missing_model_keys = [k for k in current_state.keys() if k not in state_dict]
        unexpected_model_keys = [k for k in state_dict.keys() if k not in current_state]
        if missing_model_keys or unexpected_model_keys:
            raise RuntimeError(
                "Checkpoint/model key mismatch. "
                f"missing={len(missing_model_keys)} unexpected={len(unexpected_model_keys)} "
                f"missing_sample={missing_model_keys[:8]} unexpected_sample={unexpected_model_keys[:8]}"
            )

        shape_mismatches = []
        for key, expected_tensor in current_state.items():
            loaded_tensor = state_dict[key]
            if isinstance(expected_tensor, torch.Tensor) and isinstance(loaded_tensor, torch.Tensor):
                if tuple(expected_tensor.shape) != tuple(loaded_tensor.shape):
                    shape_mismatches.append(
                        (key, tuple(loaded_tensor.shape), tuple(expected_tensor.shape))
                    )
        if shape_mismatches:
            preview = "; ".join(
                [f"{k}: {old}->{new}" for k, old, new in shape_mismatches[:8]]
            )
            raise RuntimeError(
                "Checkpoint tensor shape mismatch detected. "
                f"count={len(shape_mismatches)} details={preview}"
            )

        self.model.load_state_dict(state_dict, strict=True)

        self.model.current_phase = int(ckpt["current_phase"])
        self.model.omega = float(ckpt["omega"])
        self.global_step = int(ckpt["global_step"])
        self.current_epoch = int(ckpt["epoch"])
        self.start_batch_idx = int(ckpt["batch_idx"])
        self.total_time = float(ckpt["total_time"])
        self.session_start = time.time()

        rng_state = ckpt["rng_state"]
        if not isinstance(rng_state, torch.Tensor):
            raise RuntimeError(
                f"Checkpoint rng_state has invalid type: {type(rng_state).__name__}."
            )
        torch.set_rng_state(rng_state.cpu())

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RRA model (strict minimal pipeline).")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training batches to run in this invocation.",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Ignore existing checkpoints and start from fresh model weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global random seed.",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Run strict startup checks and exit.",
    )
    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be > 0 when provided.")

    return args


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    logger.info(
        "Initializing training: device=%s batch_size=%d seq_len=%d epochs=%d",
        DEVICE,
        BATCH_SIZE,
        SEQ_LEN,
        EPOCHS,
    )
    if args.max_steps is not None:
        logger.info("Run constraint active: max_steps=%d", args.max_steps)

    train_loader, val_loader = create_dataloaders(DATA_DIR, BATCH_SIZE, SEQ_LEN)
    logger.info(
        "Dataloaders ready: train_batches=%d val_batches=%d",
        len(train_loader),
        len(val_loader),
    )

    model = CognitiveOrganism(
        input_dim=(WORKING_DIM // 8) * C_DIM,
        vocab_size=train_loader.dataset.vocab_size,
        output_dim=8,
        device=DEVICE,
    )
    learning_brain = LearningBrain(
        L=L_DIM,
        R=R_DIM,
        D=WORKING_DIM,
        C=C_DIM,
        device=DEVICE,
    )

    if bool(_require_config("PREFLIGHT_ENABLED")) or args.preflight_only:
        run_preflight_checks(model, learning_brain, DEVICE)
        if args.preflight_only:
            logger.info("--preflight_only completed successfully.")
            return

    trainer = Trainer(model=model, learning_brain=learning_brain, device=DEVICE)

    latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if args.fresh_start:
        logger.info("Fresh start requested: checkpoint loading skipped.")
    elif trainer.load_checkpoint(latest_checkpoint):
        logger.info(
            "Checkpoint restored: epoch=%d batch_idx=%d global_step=%d omega=%.4f",
            trainer.current_epoch,
            trainer.start_batch_idx,
            trainer.global_step,
            float(model.omega),
        )
    else:
        logger.info("No checkpoint found at %s. Starting from scratch.", latest_checkpoint)

    steps_remaining = args.max_steps

    try:
        for epoch in range(trainer.current_epoch, EPOCHS):
            trainer.current_epoch = epoch
            logger.info("Starting epoch %d", epoch)

            avg_train_loss, steps_done, hit_step_limit = trainer.train_epoch(
                train_loader,
                epoch_idx=epoch,
                max_steps=steps_remaining,
            )
            if steps_remaining is not None:
                steps_remaining -= steps_done

            if hit_step_limit or (steps_remaining is not None and steps_remaining <= 0):
                logger.info("Reached --max_steps limit. Saving checkpoint and exiting.")
                trainer.save_checkpoint(latest_checkpoint, batch_idx=trainer.start_batch_idx)
                break

            val_loss = trainer.validate(val_loader, avg_train_loss)
            logger.info(
                "Epoch %d complete: train_loss=%.6f val_loss=%.6f",
                epoch,
                avg_train_loss,
                val_loss,
            )

            trainer.current_epoch = epoch + 1
            trainer.start_batch_idx = 0
            trainer.save_checkpoint(
                os.path.join(CHECKPOINT_DIR, f"train_epoch_{epoch}.pt"),
                batch_idx=0,
            )
            trainer.save_checkpoint(latest_checkpoint, batch_idx=0)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Saving latest checkpoint and exiting.")
        trainer.save_checkpoint(latest_checkpoint, batch_idx=trainer.start_batch_idx)

    logger.info("Training cycle complete.")


if __name__ == "__main__":
    main()
