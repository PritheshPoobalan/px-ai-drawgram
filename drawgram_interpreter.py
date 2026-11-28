"""CLI image interpreter for DrawGram shape-to-command drawings."""

import argparse
from typing import Dict
import cv2

from drawgram_core import ALLOWED_COMMANDS, detect_shapes, get_command


def run_commands(shapes) -> Dict[str, object]:
    """Execute recognized shape commands safely without raw exec()."""
    variables: Dict[str, object] = {}
    for shape in shapes:
        cmd = get_command(shape)
        handler = ALLOWED_COMMANDS.get(cmd) if cmd else None
        if handler:
            print(f"Running: {cmd}")
            handler(variables)

    if "x" in variables:
        print(f"Variable x = {variables['x']}")
    return variables


def main(image_path: str = "test.png") -> None:
    """Load an image file, detect shapes, and run mapped commands."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'.")
        return

    shapes = detect_shapes(img)
    run_commands(shapes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpret shapes in a DrawGram image.")
    parser.add_argument("image", nargs="?", default="test.png", help="Path to input image")
    args = parser.parse_args()
    main(args.image)
