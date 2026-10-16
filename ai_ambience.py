import colorsys
import threading
import time

import hid
import mss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

VID, PID = 0x36AE, 0xFEAB

# ---------------------------------------------------------------- capture
FPS = 20                 # lighting loop; AI runs separately in a thread
EDGE = 0.10
BOTTOM_SKIP = 0.10
SAMPLE_STEP = 6          # pixel stride for colour sampling

# --------------------------------------------------------------------- AI
MODEL_ID = "openai/clip-vit-base-patch32"
AI_INTERVAL = 0.5        # seconds between classifications (background)
PROB_SMOOTHING = 0.55    # EMA weight of the newest AI result
AI_CONFIDENCE = 0.40     # smoothed probability needed to win (6 classes)
SWITCH_MARGIN = 0.10     # new scene must beat the active one by this much
CONFIRM_SECONDS = 0.6    # how long a new scene must keep leading
DEBUG = True             # print top-3 scene probabilities (for tuning)

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# name, CLIP description, hue (deg), transition time constant (s),
# follow_screen (0..1: how much the real screen hue bends the palette)
SCENES = [
    ("AIRCRAFT", "an aircraft flying in the sky", 210.0, 2.0, 0.0),
    ("EXPLOSION/FIRE", "a big explosion with bright orange fire", 25.0, 0.35, 0.0),
    ("SMOKE", "thick grey smoke filling the view", 185.0, 1.5, 0.0),
    ("INDOOR", "an indoor room or building interior", 38.0, 2.5, 0.5),
    ("OUTDOOR", "an outdoor landscape with sky and ground", 115.0, 2.5, 0.5),
    ("DARK", "a very dark screen with almost nothing visible", 225.0, 2.5, 0.3),
]
EXPLOSION_INDEX = 1      # allowed to trigger instantly (no confirm delay)

# ----------------------------------------------------------- stable colour
MIN_SATURATION = 0.28
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 235
COLOUR_BIN_SIZE = 32

# ------------------------------------------------------------------ output
SEND_INTERVAL = 0.08
MIN_SEND_HUE_DELTA = 1.0
OUTPUT_SATURATION = 1.0
OUTPUT_BRIGHTNESS = 1.0
DEFAULT_TAU = 2.5        # used before any scene is confirmed


# ------------------------------------------------------------- keyboard
def open_keyboard():
    matches = [
        d for d in hid.enumerate(VID, PID)
        if d.get("usage_page") == 0xFF00
        and d.get("usage") == 0x02
    ]
    if not matches:
        raise RuntimeError(
            "SDCX vendor HID interface not found. "
            "Check that the keyboard is connected."
        )
    dev = hid.device()
    dev.open_path(matches[0]["path"])
    return dev


def set_colour(dev, rgb):
    r, g, b = np.clip(rgb, 0, 255) / 255.0
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hsv = (
        round(hue * 255) % 256,
        round(sat * 255),
        round(val * 255),
    )
    payload = bytes([
        0x06, 0x0B, 0x0B,
        0x00, 0x00,
        0x01, 0x00,
        0x01,       # Static mode
        0x04,       # Maximum hardware brightness
        0x02,
        0x01,
        0xFF,       # Confirmed colour flag
        0x00,
        *hsv,
    ])
    dev.write(bytes([0x00]) + payload.ljust(64, b"\x00"))


# ------------------------------------------------------------------- AI
class SceneAI(threading.Thread):
    """Runs CLIP in the background so lighting never stalls on inference."""

    def __init__(self):
        super().__init__(daemon=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading local scene model on {self.device.upper()}...")

        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(MODEL_ID)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Raw cosine similarities are tiny and nearly equal, so softmax on
        # them alone is ~uniform. CLIP's learned logit scale (~100) fixes it.
        self.logit_scale = float(self.model.logit_scale.exp().item())

        prompts = [f"a screenshot of a video game showing {s[1]}."
                   for s in SCENES]
        text = processor(text=prompts, return_tensors="pt", padding=True)
        text = {k: v.to(self.device) for k, v in text.items()}
        with torch.inference_mode():
            tf = self._text_features(text)
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)

        n = len(SCENES)
        self.probs = np.full(n, 1.0 / n, dtype=np.float32)
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # get_*_features returns a tensor in older transformers but a
    # BaseModelOutputWithPooling in newer ones. Going through the towers and
    # projections directly works on both.
    def _text_features(self, text):
        out = self.model.text_model(
            input_ids=text["input_ids"],
            attention_mask=text.get("attention_mask"),
        )
        return self.model.text_projection(out.pooler_output)

    def _image_features(self, pixel_values):
        out = self.model.vision_model(pixel_values=pixel_values)
        return self.model.visual_projection(out.pooler_output)

    def submit(self, rgb_small):
        with self._lock:
            self._frame = rgb_small

    def get_probs(self):
        with self._lock:
            return self.probs.copy()

    def stop(self):
        self._stop.set()

    def _prepare(self, rgb_small):
        # Squash the whole frame to 224x224 (no centre crop), so the
        # edges of a widescreen image still reach the model.
        img = Image.fromarray(rgb_small).resize((224, 224), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - CLIP_MEAN) / CLIP_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype)

    def run(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.02)
                continue

            started = time.monotonic()
            pixel_values = self._prepare(frame)
            with torch.inference_mode():
                f = self._image_features(pixel_values)
                f = f / f.norm(dim=-1, keepdim=True)
                logits = self.logit_scale * (f @ self.text_features.T)
                p = logits.float().softmax(dim=-1)[0].cpu().numpy()

            with self._lock:
                self.probs = (
                    PROB_SMOOTHING * p
                    + (1.0 - PROB_SMOOTHING) * self.probs
                )

            spent = time.monotonic() - started
            time.sleep(max(0.0, AI_INTERVAL - spent))


# ---------------------------------------------------------- screen colour
_mask_cache = {}


def edge_mask(shape):
    if shape not in _mask_cache:
        h, w = shape
        ex = max(1, int(w * EDGE))
        ey = max(1, int(h * EDGE))
        bottom = int(h * (1 - BOTTOM_SKIP))
        m = np.zeros((h, w), dtype=bool)
        m[:ey, :] = True
        m[:bottom, :ex] = True
        m[:bottom, w - ex:] = True
        m[max(0, bottom - ey):bottom, :] = True
        _mask_cache[shape] = m
    return _mask_cache[shape]


def edge_pixels(frame):
    sub = frame[::SAMPLE_STEP, ::SAMPLE_STEP]
    return sub[edge_mask(sub.shape[:2])].astype(np.float32)


def dominant_colour(pixels):
    if len(pixels) == 0:
        return None
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    saturation = np.divide(
        maximum - minimum, maximum,
        out=np.zeros_like(maximum), where=maximum > 0,
    )
    valid = (
        (saturation >= MIN_SATURATION)
        & (maximum >= MIN_BRIGHTNESS)
        & (maximum <= MAX_BRIGHTNESS)
    )
    colourful = pixels[valid]
    if len(colourful) < 8:
        return None

    bins = (colourful // COLOUR_BIN_SIZE).astype(np.uint8)
    packed = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    counts = np.bincount(packed, minlength=512)
    return np.median(colourful[packed == int(counts.argmax())], axis=0)


# ------------------------------------------------------------- hue maths
def hue_from_rgb(rgb):
    r, g, b = np.clip(rgb, 0, 255) / 255.0
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def hue_distance(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def move_hue(current, target, amount):
    """Move around the hue circle (handles the 359 -> 0 wrap)."""
    delta = (target - current + 180.0) % 360.0 - 180.0
    return (current + amount * delta) % 360.0


def bright_rgb(hue):
    r, g, b = colorsys.hsv_to_rgb(
        (hue % 360.0) / 360.0, OUTPUT_SATURATION, OUTPUT_BRIGHTNESS
    )
    return tuple(round(c * 255) for c in (r, g, b))


# ------------------------------------------------------------ scene logic
class SceneTracker:
    """Hysteresis + dwell time so the active scene doesn't flicker."""

    def __init__(self):
        self.active = None
        self.candidate = None
        self.candidate_since = 0.0

    def update(self, probs, now):
        top = int(probs.argmax())
        if probs[top] < AI_CONFIDENCE:
            self.candidate = None
            return self.active

        if self.active is None:
            self.active = top
            return self.active

        if top == self.active:
            self.candidate = None
            return self.active

        if probs[top] - probs[self.active] < SWITCH_MARGIN:
            self.candidate = None
            return self.active

        if top != self.candidate:
            self.candidate = top
            self.candidate_since = now

        dwell = 0.0 if top == EXPLOSION_INDEX else CONFIRM_SECONDS
        if now - self.candidate_since >= dwell:
            self.active = top
            self.candidate = None
        return self.active


# ------------------------------------------------------------------ main
def main():
    keyboard = None
    ai = None

    try:
        ai = SceneAI()           # load the model before opening the keyboard
        keyboard = open_keyboard()
        ai.start()

        tracker = SceneTracker()
        current_hue = None
        last_sent_hue = None
        last_send = 0.0
        last_log = 0.0
        previous = time.monotonic()

        print("LumaSync AI Ambient — Ctrl+C to stop")

        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                raise RuntimeError("No physical display detected.")
            monitor = sct.monitors[1]

            while True:
                loop_start = time.monotonic()
                dt = min(loop_start - previous, 0.25)
                previous = loop_start

                # One capture per loop, reused for colour and AI.
                frame = np.ascontiguousarray(
                    np.asarray(sct.grab(monitor))[:, :, :3][:, :, ::-1]
                )
                ai.submit(np.ascontiguousarray(frame[::4, ::4]))

                dominant = dominant_colour(edge_pixels(frame))
                screen_hue = (
                    hue_from_rgb(dominant) if dominant is not None else None
                )

                probs = ai.get_probs()
                active = tracker.update(probs, loop_start)

                # Choose the target hue (never accumulates drift).
                target_hue = None
                tau = DEFAULT_TAU
                if active is not None:
                    _, _, base_hue, tau, follow = SCENES[active]
                    target_hue = base_hue
                    if follow > 0 and screen_hue is not None:
                        target_hue = move_hue(base_hue, screen_hue, follow)
                elif screen_hue is not None:
                    target_hue = screen_hue

                if target_hue is not None:
                    if current_hue is None:
                        current_hue = target_hue
                    alpha = 1.0 - np.exp(-dt / tau)
                    current_hue = move_hue(current_hue, target_hue, alpha)

                    now = time.monotonic()
                    changed = (
                        last_sent_hue is None
                        or hue_distance(current_hue, last_sent_hue)
                        >= MIN_SEND_HUE_DELTA
                    )
                    if (changed or now - last_send >= 1.0) \
                            and now - last_send >= SEND_INTERVAL:
                        set_colour(keyboard, bright_rgb(current_hue))
                        last_sent_hue = current_hue
                        last_send = now

                    if now - last_log >= 0.5:
                        name = SCENES[active][0] if active is not None \
                            else "WAITING"
                        line = f"\rSCENE={name:<15} HUE={current_hue:5.0f}"
                        if DEBUG:
                            order = np.argsort(probs)[::-1][:3]
                            top3 = " ".join(
                                f"{SCENES[i][0][:4]}:{probs[i]:.2f}"
                                for i in order
                            )
                            line += f"  [{top3}]"
                        print(line + "      ", end="", flush=True)
                        last_log = now

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, 1.0 / FPS - elapsed))

    except KeyboardInterrupt:
        print("\nLumaSync stopped.")
    finally:
        if ai is not None:
            ai.stop()
        if keyboard is not None:
            keyboard.close()


if __name__ == "__main__":
    main()