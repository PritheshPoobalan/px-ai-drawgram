"""Interactive OpenCV drawing window for DrawGram."""

from typing import Dict, Tuple
import cv2
import numpy as np

from drawgram_core import detect_shapes, get_live_command


CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700
WINDOW_NAME = "DrawGram Live"
STROKE_COLOR_BGR = (0, 0, 0)
STROKE_THICKNESS = 3


def make_blank_canvas(height: int = CANVAS_HEIGHT, width: int = CANVAS_WIDTH) -> np.ndarray:
    """Return a clean white BGR drawing canvas."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


class DrawingState:
    """Track mouse stroke state and canvas buffer."""

    def __init__(self, height: int = CANVAS_HEIGHT, width: int = CANVAS_WIDTH) -> None:
        self.height = height
        self.width = width
        self.drawing = False
        self.prev_pt: Tuple[int, int] = (-1, -1)
        self.canvas = make_blank_canvas(height, width)

    def clear(self) -> None:
        """Reset stroke state and clear canvas to white."""
        self.drawing = False
        self.prev_pt = (-1, -1)
        self.canvas = make_blank_canvas(self.height, self.width)


def _draw_segment(canvas: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    """Draw a stroke segment when coordinates are valid."""
    if start[0] >= 0 and start[1] >= 0:
        cv2.line(canvas, start, end, STROKE_COLOR_BGR, thickness=STROKE_THICKNESS)


def on_mouse(event: int, x: int, y: int, flags: int, state: DrawingState) -> None:
    """Handle mouse events on the live canvas."""
    _ = flags
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.prev_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        curr_pt = (x, y)
        _draw_segment(state.canvas, state.prev_pt, curr_pt)
        state.prev_pt = curr_pt
    elif event == cv2.EVENT_LBUTTONUP:
        if state.drawing:
            _draw_segment(state.canvas, state.prev_pt, (x, y))
        state.drawing = False
        state.prev_pt = (-1, -1)


def detect_and_run(img: np.ndarray) -> Dict[str, object]:
    """Detect shapes top-to-bottom and execute mapped live actions."""
    shapes = detect_shapes(img)
    variables: Dict[str, object] = {}

    for shape in shapes:
        shape_type = shape.get("type") if isinstance(shape, dict) else None
        shape_color = shape.get("color") if isinstance(shape, dict) else None
        action = get_live_command(shape_type, shape_color)
        if not action:
            continue

        op, val = action
        if op == "print":
            print("Running: print(val)")
            print(val)
        elif op == "set_x":
            print(f"Running: x = {val!r}")
            variables["x"] = val
        elif op == "add_x" and isinstance(variables.get("x"), (int, float)) and isinstance(val, (int, float)):
            variables["x"] = variables["x"] + val
            print(f"Running: x += {val!r} -> {variables['x']!r}")

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
