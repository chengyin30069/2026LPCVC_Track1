from __future__ import annotations

from distill import parse_args, run_training


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
