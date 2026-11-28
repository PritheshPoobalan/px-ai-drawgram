"""Interactive OpenCV drawing window for DrawGram."""

from typing import Dict, Tuple
import cv2
import numpy as np

from drawgram_core import detect_shapes, get_live_command


CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700
WINDOW_NAME = "DrawGram Live"


def make_blank_canvas() -> np.ndarray:
    """Return a clean white drawing canvas."""
    return np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255


class DrawingState:
    """Track mouse stroke state and canvas buffer."""

    def __init__(self) -> None:
        self.drawing = False
        self.prev_pt: Tuple[int, int] = (-1, -1)
        self.canvas = make_blank_canvas()

    def clear(self) -> None:
        self.canvas = make_blank_canvas()


def on_mouse(event: int, x: int, y: int, flags: int, state: DrawingState) -> None:
    """Handle mouse events on the live canvas."""
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.prev_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        cv2.line(state.canvas, state.prev_pt, (x, y), (0, 0, 0), thickness=3)
        state.prev_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        cv2.line(state.canvas, state.prev_pt, (x, y), (0, 0, 0), thickness=3)


def detect_and_run(img: np.ndarray) -> Dict[str, object]:
    """Detect shapes top-to-bottom and execute mapped live actions."""
    shapes = detect_shapes(img)
    variables: Dict[str, object] = {}

    for shape in shapes:
        action = get_live_command(shape["type"], shape["color"])
        if not action:
            continue
        op, val = action
        if op == "print":
            print("Running: print('Hello from Live Draw!')")
            print(val)
        elif op == "set_x":
            print("Running: x = 42")
            variables["x"] = val

    if "x" in variables:
        print(f"Variable x = {variables['x']}")
    return variables


def main() -> None:
    """Run the interactive DrawGram window loop."""
    state = DrawingState()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    print("Draw on the window. Press 'r' to run, 'c' to clear, 'q' to quit.")

    while True:
        cv2.imshow(WINDOW_NAME, state.canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            print("\n--- Running DrawGram Interpreter ---")
            detect_and_run(state.canvas)
            print("--- Done ---\n")
        elif key == ord("c"):
            state.clear()
            print("Canvas cleared.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
