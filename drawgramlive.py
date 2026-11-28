import cv2
import numpy as np


CANVAS_HEIGHT = 500
CANVAS_WIDTH = 700
WINDOW_NAME = "DrawGram Live"


def make_blank_canvas():
    """Return a clean white drawing canvas."""
    return np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255


drawing = False
ix, iy = -1, -1
canvas = make_blank_canvas()


def draw(event, x, y, flags, param):
    """Handle mouse drawing events on the live canvas."""
    global ix, iy, drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (ix, iy), (x, y), (0, 0, 0), thickness=3)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), (0, 0, 0), thickness=3)


def get_live_command(shape_type, color):
    """Map detected shape and BGR color on the live canvas to a safe action."""
    if not np.array_equal(color, np.array([0, 0, 0])):
        return None
    if shape_type == "circle":
        return "print", "Hello from Live Draw!"
    if shape_type == "square":
        return "set_x", 42
    return None


def detect_and_run(img):
    """Detect shapes top-to-bottom and execute mapped live actions safely."""
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

    shapes.sort(key=lambda s: s["pos"][1])
    variables = {}

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
        print("Variable x =", variables["x"])


def main():
    global canvas
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, draw)

    print("Draw on the window. Press 'r' to run, 'c' to clear, 'q' to quit.")

    while True:
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            print("\n--- Running DrawGram Interpreter ---")
            detect_and_run(canvas)
            print("--- Done ---\n")
        elif key == ord("c"):
            canvas = make_blank_canvas()
            print("Canvas cleared.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
