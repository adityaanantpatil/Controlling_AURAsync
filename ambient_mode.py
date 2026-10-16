
import colorsys
import time

import hid
import mss
import numpy as np

VID, PID = 0x36AE, 0xFEAB

# Capture and update
FPS = 10
SEND_INTERVAL = 0.10

# Colour stability
MIN_SATURATION = 0.28
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 235
COLOUR_BIN_SIZE = 32
SWITCH_CONFIRM_FRAMES = 2
HUE_SMOOTHING = 0.45
MIN_HUE_CHANGE = 4.0

# Screen sampling
EDGE = 0.10
BOTTOM_SKIP = 0.10


def open_keyboard():
    matches = [
        d for d in hid.enumerate(VID, PID)
        if d.get("usage_page") == 0xFF00
        and d.get("usage") == 0x02
    ]

    if not matches:
        raise RuntimeError("SDCX vendor HID interface not found.")

    dev = hid.device()
    dev.open_path(matches[0]["path"])
    return dev


def get_edge_pixels(sct, monitor):
    # MSS returns BGRA; convert to RGB.
    frame = np.asarray(sct.grab(monitor))[:, :, :3][:, :, ::-1]
    height, width, _ = frame.shape

    ex = max(1, int(width * EDGE))
    ey = max(1, int(height * EDGE))
    bottom = int(height * (1 - BOTTOM_SKIP))

    mask = np.zeros((height, width), dtype=bool)
    mask[:ey, :] = True
    mask[:bottom, :ex] = True
    mask[:bottom, width - ex:] = True
    mask[max(0, bottom - ey):bottom, :] = True

    # Downsample for performance.
    pixels = frame[::6, ::6]
    pixels = pixels[mask[::6, ::6]]

    return pixels.astype(np.float32)


def dominant_colour(pixels):
    """Return the most common meaningful chromatic colour, or None."""
    if len(pixels) == 0:
        return None

    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)

    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )

    # Ignore near-black, near-white and neutral pixels.
    valid = (
        (saturation >= MIN_SATURATION)
        & (maximum >= MIN_BRIGHTNESS)
        & (maximum <= MAX_BRIGHTNESS)
    )
    colourful = pixels[valid]

    # If the frame is mostly black/white/grey, keep the previous colour.
    if len(colourful) < 8:
        return None

    bins = (colourful // COLOUR_BIN_SIZE).astype(np.uint8)

    # Eight bins per channel -> 512 possible buckets.
    packed = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    counts = np.bincount(packed, minlength=512)

    dominant_bin = int(counts.argmax())
    selected = colourful[packed == dominant_bin]

    return np.median(selected, axis=0)


def rgb_to_hue(rgb):
    r, g, b = np.clip(rgb, 0, 255) / 255.0
    hue, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return hue * 360.0


def hue_distance(a, b):
    """Shortest distance between two hue angles."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def smooth_hue(current, target, amount):
    """Interpolate around the hue circle, including red across 360/0."""
    delta = (target - current + 180.0) % 360.0 - 180.0
    return (current + amount * delta) % 360.0


def bright_rgb_from_hue(hue):
    """Always produce a vivid, maximum-brightness RGB colour."""
    r, g, b = colorsys.hsv_to_rgb(
        (hue % 360.0) / 360.0,
        1.0,
        1.0,
    )
    return tuple(round(c * 255) for c in (r, g, b))


def set_colour(dev, rgb):
    # Convert the final vivid RGB colour to the keyboard's HSV protocol.
    r, g, b = np.clip(rgb, 0, 255) / 255.0
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)

    hsv = (
        round(hue * 255) % 256,
        round(sat * 255),
        round(val * 255),
    )

    # Confirmed SDCX static-colour packet.
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


def main():
    keyboard = open_keyboard()

    current_hue = None
    candidate_hue = None
    candidate_frames = 0
    last_sent_hue = None
    last_send_time = 0.0

    try:
        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                raise RuntimeError("No physical display detected.")

            # First physical display: laptop screen.
            monitor = sct.monitors[1]

            print("LumaSync Ambient — Ctrl+C to stop")
            print("Sampling screen edges; output stays bright.")

            while True:
                started = time.perf_counter()

                pixels = get_edge_pixels(sct, monitor)
                detected = dominant_colour(pixels)
                now = time.monotonic()

                if detected is not None:
                    target_hue = rgb_to_hue(detected)

                    if current_hue is None:
                        # Initialise immediately.
                        current_hue = target_hue
                        candidate_hue = None
                        candidate_frames = 0

                    elif hue_distance(target_hue, current_hue) < MIN_HUE_CHANGE:
                        # Small changes are noise; keep the current hue.
                        candidate_hue = None
                        candidate_frames = 0

                    else:
                        # Require a consistent new colour for two frames.
                        if (
                            candidate_hue is not None
                            and hue_distance(target_hue, candidate_hue)
                            < 25.0
                        ):
                            candidate_hue = (
                                candidate_hue + target_hue
                            ) / 2.0
                            candidate_frames += 1
                        else:
                            candidate_hue = target_hue
                            candidate_frames = 1

                        if candidate_frames >= SWITCH_CONFIRM_FRAMES:
                            # Confirmed scene change; transition quickly.
                            current_hue = smooth_hue(
                                current_hue,
                                candidate_hue,
                                HUE_SMOOTHING,
                            )
                            candidate_hue = None
                            candidate_frames = 0

                if current_hue is not None:
                    # Continue smoothing toward a confirmed candidate
                    # on subsequent frames without dimming the output.
                    if candidate_hue is None:
                        output = bright_rgb_from_hue(current_hue)
                    else:
                        output = bright_rgb_from_hue(current_hue)

                    should_send = (
                        last_sent_hue is None
                        or hue_distance(current_hue, last_sent_hue) >= 2.0
                        or now - last_send_time >= 1.0
                    )

                    if should_send and now - last_send_time >= SEND_INTERVAL:
                        set_colour(keyboard, output)
                        last_sent_hue = current_hue
                        last_send_time = now

                    detected_text = (
                        tuple(detected.astype(int))
                        if detected is not None
                        else "neutral; holding hue"
                    )

                    print(
                        f"\rDETECTED={detected_text} "
                        f"HUE={current_hue:.0f} "
                        f"OUTPUT={output}       ",
                        end="",
                        flush=True,
                    )

                elapsed = time.perf_counter() - started
                time.sleep(max(0, 1 / FPS - elapsed))

    except KeyboardInterrupt:
        print("\nAURASync stopped.")
    finally:
        keyboard.close()


if __name__ == "__main__":
    main()
