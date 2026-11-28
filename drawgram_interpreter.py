import cv2
import numpy as np


ALLOWED_COMMANDS = {
    "print('Hello World!')": lambda ns: print("Hello World!"),
    "x = 1": lambda ns: ns.__setitem__("x", 1),
}


def get_command(shape):
    """Map a detected shape and BGR color to a safe, known command string."""
    color = shape.get("color")
    shape_type = shape.get("type")
    if color is None or shape_type is None:
        return None

    # OpenCV loads images in BGR format.
    if shape_type == "circle" and np.array_equal(color, np.array([255, 255, 255])):
        return "print('Hello World!')"
    if shape_type == "square" and np.array_equal(color, np.array([0, 0, 255])):
        return "x = 1"
    return None


def detect_shapes(img):
    """Detect basic geometric shapes and their center BGR colors from an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[:2]
    shapes = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 25:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = min(max(x + w // 2, 0), width - 1)
        cy = min(max(y + h // 2, 0), height - 1)
        color = img[cy, cx]

        vertices = len(approx)
        if vertices == 3:
            shape_type = "triangle"
        elif vertices == 4:
            shape_type = "square"
        elif vertices > 4:
            shape_type = "circle"
        else:
            shape_type = None

        if shape_type:
            shapes.append({"type": shape_type, "pos": (cx, cy), "color": color})

    return sorted(shapes, key=lambda s: s["pos"][1])


def run_commands(shapes):
    """Execute recognized shape commands safely without raw exec()."""
    variables = {}
    for shape in shapes:
        cmd = get_command(shape)
        handler = ALLOWED_COMMANDS.get(cmd) if cmd else None
        if handler:
            print("Running:", cmd)
            handler(variables)

    if "x" in variables:
        print("Variable x =", variables["x"])
    return variables


def main(image_path="test.png"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image '{image_path}' not found.")
        return

    shapes = detect_shapes(img)
    run_commands(shapes)


if __name__ == "__main__":
    main()
