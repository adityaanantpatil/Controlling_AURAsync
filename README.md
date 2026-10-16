# AURAsync: Screen-Reactive Ambient Lighting

Screen-reactive ambient lighting for the **SDCX SDMulti** keyboard (USB VID `0x36AE`, PID `0xFEAB`), controlled directly over USB HID with no manufacturer RGB software.

> **This is a later stage of AURAsync.** The original repository, **[AURAsync](https://github.com/adityaanantpatil/AURAsync)**, covers the earlier work: reverse-engineering the keyboard's vendor HID protocol (Wireshark captures, lighting read/write commands, and Python scripts to enumerate the device, read its configuration and set colours). If you want the protocol write-up and the low-level scripts, start there. This stage builds on them and assumes the packet format documented in that repository.

Where the earlier stage achieved reliable static colour control, this stage turns the keyboard into an ambient light that follows what is on your screen while gaming, watching movies, streaming, or just using the desktop.

## Project stages

| Stage | What it covers | Where |
|-------|----------------|-------|
| 1. Protocol and static control | HID reverse engineering, reading and setting colours | [AURAsync repository](https://github.com/adityaanantpatil/AURAsync) |
| 2. Ambient and AI lighting (this document) | Screen-edge colour following and experimental AI scene recognition | This README |

## Status

| Component | State |
|-----------|-------|
| Static colour control over HID | Working (from AURAsync) |
| `ambient_mode.py` (screen-edge colour) | Working, correct colours; transitions still being smoothed |
| `ai_ambience.py` (CLIP scene recognition) | Experimental prototype, not validated |

Known issue under investigation: the keyboard appears to blank briefly on some colour updates, which makes fast transitions look harsh or flickery. See [Known issues](#known-issues).

## How it works

Both modes share the same output path:

1. Capture the first physical display with [`mss`](https://github.com/BoboTiG/python-mss).
2. Work out a target **hue** from the screen.
3. Ease the current hue toward the target over time.
4. Send the result to the keyboard as a static, full-brightness, full-saturation colour using the confirmed SDCX static-colour packet (HSV encoded in the last three payload bytes).

Output is always vivid and at maximum brightness. Only the hue changes.

### Ambient mode (`ambient_mode.py`)

Follows the colours at the edges of the screen, similar to a TV backlight.

- **Sampling.** Only a border around the screen (10% on the top and sides, and a band above the bottom 10%, which is skipped to avoid taskbars and HUD clutter) is analysed. The frame is subsampled for speed and the edge mask is cached.
- **Colour detection.** Near-black, near-white and neutral pixels are ignored. Each remaining pixel votes for a 10-degree hue bin, weighted by saturation and brightness. The histogram is smoothed around the colour circle and averaged over a short window, so single-frame flashes do not change the colour.
- **Stability.** A new hue peak must be clearly stronger than the current one before it takes over. If a frame is mostly grey, black or white, the last colour is held.
- **Motion.** The output hue eases toward the target every frame with a time constant, so transitions do not depend on frame rate and do not move in steps. Hue interpolation goes the short way around the colour wheel, including across red (360/0).

### AI mode (`ai_ambience.py`)

An experimental layer that recognises the type of scene and picks a palette for it.

- **Model.** Uses OpenAI's CLIP (`openai/clip-vit-base-patch32`) locally through Hugging Face `transformers`, on the GPU when CUDA is available. No API keys and nothing leaves the machine.
- **Zero-shot scenes.** The screen is compared with text descriptions of scenes such as an aircraft, an explosion with fire, thick smoke, an indoor space, an outdoor landscape and a very dark screen. Each scene has its own palette hue and transition speed (explosions react quickly, ambient scenes fade slowly).
- **Background inference.** Classification runs in a separate thread so lighting updates never wait on the model.
- **Smoothing.** Probabilities are averaged over time, a new scene must beat the current one by a margin and stay ahead briefly, and explosions are allowed to trigger immediately.
- **Neutral scenes** (indoor, outdoor, dark) blend their palette hue with the actual screen colour.

This mode is a proof of concept. Recognition accuracy, real-time behaviour and palette choices have not been validated, and in early tests it showed mostly blue shades and flickered. The current plan is to keep ambient mode as the base and use the AI only to add events (for example explosion or fire) on top of it.

## Requirements

- Windows (developed and tested on Windows 10/11 with Python 3.10)
- SDCX SDMulti keyboard connected over USB
- Python 3.10+
- Optional for AI mode: an NVIDIA GPU with a CUDA-enabled PyTorch build (CPU works but is slower)

```bash
pip install hidapi mss numpy
# AI mode only
pip install torch transformers pillow
```

The first run of AI mode downloads the CLIP model (about 600 MB) from the Hugging Face Hub. The warnings about unauthenticated requests and symlinks on Windows are harmless.

## Usage

Close any manufacturer RGB software first, since it may hold the HID interface.

```bash
# Screen-edge ambient lighting
python ambient_mode.py

# AI scene recognition (experimental)
python ai_ambience.py
```

Stop either script with `Ctrl+C`.

Both scripts use the first physical display (`mss` monitor index 1). Change `monitor = sct.monitors[1]` if you want a different screen.

## Tuning

### `ambient_mode.py`

| Constant | Effect |
|----------|--------|
| `HUE_TAU` | Seconds to ease toward the target hue. Lower is snappier, higher is more cinematic. |
| `HIST_TAU` | How long colour evidence is averaged. Lower reacts faster but is more sensitive to flashes. |
| `PEAK_SWITCH_RATIO` | How much stronger a new colour must be before it replaces the current one. |
| `MIN_SATURATION`, `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS` | Which pixels count as meaningful colour. |
| `MIN_COVERAGE` | Share of edge pixels that must be colourful, otherwise the hue is held. |
| `EDGE`, `BOTTOM_SKIP` | Size of the sampled border and the skipped bottom strip. |
| `SEND_INTERVAL`, `MIN_SEND_HUE_DELTA` | How often and how finely colour packets are sent. |

### `ai_ambience.py`

| Constant | Effect |
|----------|--------|
| `SCENES` | Scene descriptions, palette hue, transition time and how much the screen colour bends the palette. |
| `AI_INTERVAL` | Seconds between classifications. |
| `AI_CONFIDENCE`, `SWITCH_MARGIN`, `CONFIRM_SECONDS` | How sure and how stable a prediction must be before the scene changes. |
| `DEBUG` | Prints the top three scene probabilities, which is the best way to tune prompts and thresholds. |

## Known issues

- **Black blinks on colour changes.** The keyboard seems to blank briefly when it receives updates, so fast fades look harsh. The static-colour packet sets the whole lighting mode, so the firmware may re-initialise on every write. A diagnostic script, `sweep_test.py`, compares isolated writes with fades at different update rates. If every packet blinks, the likely fix is to find a lighter "set colour only" packet in a Wireshark capture of the manufacturer software while dragging a colour slider.
- **AI mode is unreliable.** Scene categories overlap (aircraft and outdoor both look like blue sky), and the fixed palette leans blue. Prompts, categories and palettes need rework.
- **Single display.** Only one monitor is sampled.
- **Laptop screen capture only.** Edge sampling does not account for letterboxing in films.

## Roadmap

- [ ] Make colour transitions smooth on the hardware (rate limiting or a better packet)
- [ ] Hybrid mode: ambient colour as the base, AI only for events such as explosions, fire and smoke
- [ ] Letterbox detection for movies
- [ ] Per-game or per-app profiles
- [ ] Config file instead of editing constants
- [ ] Multi-monitor selection

## Project structure

```
ambient_mode.py   Screen-edge ambient lighting
ai_ambience.py    CLIP-based scene recognition (experimental)
sweep_test.py     Diagnostic for keyboard blanking during colour updates
```

The protocol notes and low-level HID scripts live in the original [AURAsync](https://github.com/adityaanantpatil/AURAsync) repository.

## Disclaimer

This project is unofficial and not affiliated with the keyboard's manufacturer. It sends custom HID packets to a vendor interface that was identified experimentally. Use it at your own risk.
