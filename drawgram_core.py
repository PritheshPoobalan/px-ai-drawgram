"""Shared shape detection and safe command mapping utilities for DrawGram."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


ALLOWED_COMMANDS = {
    "print('Hello World!')": lambda ns: print("Hello World!"),
    "x = 1": lambda ns: ns.__setitem__("x", 1),
}


def detect_shapes(
    img: np.ndarray,
    min_area: float = 25.0,
    canny_low: int = 50,
    canny_high: int = 150,
) -> List[Dict[str, Any]]:
    """Detect triangles, squares, and circles along with center coordinates and BGR color."""
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[:2]
    shapes: List[Dict[str, Any]] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = min(max(x + w // 2, 0), width - 1)
        cy = min(max(y + h // 2, 0), height - 1)
        color = img[cy, cx]

        vertices = len(approx)
        if vertices == 3:
            shape_type: Optional[str] = "triangle"
        elif vertices == 4:
            shape_type = "square"
        elif vertices > 4:
            shape_type = "circle"
        else:
            shape_type = None

        if shape_type:
            shapes.append({"type": shape_type, "pos": (cx, cy), "color": color})

    return sorted(shapes, key=lambda s: s["pos"][1])


def get_command(shape: Dict[str, Any]) -> Optional[str]:
    """Map a detected shape and BGR color tuple to an allowed command string."""
    color = shape.get("color")
    shape_type = shape.get("type")
    if color is None or shape_type is None:
        return None

    bgr = tuple(int(v) for v in color[:3]) if hasattr(color, "__len__") else color
    if shape_type == "circle" and bgr == (255, 255, 255):
        return "print('Hello World!')"
    if shape_type == "square" and bgr == (0, 0, 255):
        return "x = 1"
    return None


def get_live_command(shape_type: Optional[str], color: Any) -> Optional[Tuple[str, Any]]:
    """Map black-ink shapes on the live canvas to safe actions."""
    if color is None or shape_type is None:
        return None
    bgr = tuple(int(v) for v in color[:3]) if hasattr(color, "__len__") else color
    if bgr != (0, 0, 0):
        return None
    if shape_type == "circle":
        return ("print", "Hello from Live Draw!")
    if shape_type == "square":
        return ("set_x", 42)
    return None
