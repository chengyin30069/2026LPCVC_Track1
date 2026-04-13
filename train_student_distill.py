from __future__ import annotations

import os
import sys


def _extract_device_spec(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--device" and index + 1 < len(argv):
            return argv[index + 1].strip()
        if token.startswith("--device="):
            return token.split("=", 1)[1].strip()
    return None


def _normalize_visible_cuda_devices(argv: list[str]) -> list[str]:
    device_spec = _extract_device_spec(argv)
    if not device_spec:
        return argv

    requested_parts = [segment.strip() for segment in device_spec.split(",") if segment.strip()]
    if not requested_parts or not requested_parts[0].startswith("cuda"):
        return argv

    physical_ids: list[int] = []
    for part in requested_parts:
        if not part.startswith("cuda"):
            raise ValueError(f"Mixed non-cuda device in --device: {device_spec}")
        if ":" in part:
            physical_ids.append(int(part.split(":", 1)[1]))
        else:
            physical_ids.append(0)

    unique_ids: list[int] = []
    for gpu_id in physical_ids:
        if gpu_id not in unique_ids:
            unique_ids.append(gpu_id)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in unique_ids)

    remapped_parts = [f"cuda:{idx}" for idx in range(len(unique_ids))]
    remapped_device_spec = ",".join(remapped_parts)

    normalized_argv = list(argv)
    for index, token in enumerate(normalized_argv):
        if token == "--device" and index + 1 < len(normalized_argv):
            normalized_argv[index + 1] = remapped_device_spec
            break
        if token.startswith("--device="):
            normalized_argv[index] = f"--device={remapped_device_spec}"
            break
    return normalized_argv


def main() -> None:
    sys.argv = _normalize_visible_cuda_devices(sys.argv)

    from distill import parse_args, run_training

    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user. Cleanly shutting down training workers.")
        raise SystemExit(130)
