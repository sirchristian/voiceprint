"""Allow running as `python -m voiceprint [train|eval]`.

Defaults to `train` for backward compatibility.
"""

import sys

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "train":
        from voiceprint.train import main as train_main
        train_main()
    elif sys.argv[1] == "eval":
        from voiceprint.eval import main as eval_main
        eval_main()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: python -m voiceprint [train|eval]")
        sys.exit(1)


if __name__ == "__main__":
    main()
