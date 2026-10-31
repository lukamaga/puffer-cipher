# Hardware build guide

The software runs on any laptop with a webcam (or with the built-in
simulator, no hardware at all). This page describes the reference rig:
a Raspberry Pi watching a real aquarium day and night, with a
server-controlled LED that proves the video is live.

## Bill of materials

| part | why | notes |
|---|---|---|
| Raspberry Pi 5 (4 GB) or Pi 4 | runs the engine, API and dashboard | ~25-30 fps tracking at 640x480 with the motion detector; YOLO needs the Pi 5 or an accelerator |
| Raspberry Pi Camera Module 3 **NoIR** (or any UVC webcam without IR-cut filter) | one sensor for day *and* night | NoIR has no infrared-cut filter, so it sees an 850 nm illuminator |
| 850 nm IR illuminator (small LED panel) | night mode | fish perceive 850 nm poorly, so it does not disturb their sleep |
| 5 mm LED (red, or 850 nm IR) + 220 Ω resistor | liveness challenge | driven from GPIO17 |
| waterproof enclosure / mount | keep electronics out of the water | mount the LED **outside the glass** facing in, inside the camera's field of view |

## LED wiring

```
 Pi GPIO17 (pin 11) ──[ 220 Ω ]──►|── GND (pin 9)
                                 LED
```

Resistor: `(3.3 V − V_f) / I`. A red LED (V_f ≈ 2.0 V) at ~6 mA or an 850 nm IR
LED (V_f ≈ 1.5 V) at ~8 mA both work with 220 Ω and stay well below the
16 mA per-pin limit. Glue the LED to the outside of the back or side glass
so it is visible in a corner of the frame; no electronics touch the water.

Configure it with `PUFFER_LED=gpio:17` (uses `gpiozero`, install with
`pip install "puffer-cipher[pi]"`). If the LED's pixel region is known, set
`PUFFER_LED_ROI="x,y,w,h"`; otherwise the verifier finds it automatically by
correlating every image region with the commanded blink pattern.

## Camera settings that preserve entropy

The sensor channel harvests *temporal pixel noise*. Camera pipelines try hard
to remove exactly that, so:

* **Turn off denoising** (`NoiseReductionMode: Off` in picamera2, or the
  webcam's "noise reduction"/"3D NR" controls). Denoise can cut the measured
  min-entropy per sample several-fold.
* **Fix exposure and gain** after the scene is lit. Auto-exposure makes noise
  statistics drift and can trip the adaptive-proportion health test.
* Prefer raw/YUV capture over MJPEG. JPEG quantisation discards the
  low-order bits where most of the sensor noise lives. The analysis report
  shows the effect (`puffer analyze` with and without `jpeg_quality`).
* At night the IR illuminator plus higher analogue gain *raises* sensor noise
  while sleeping fish contribute less motion entropy. The engine re-weights
  automatically because credit is always computed from measured min-entropy.

## Raspberry Pi setup

```bash
sudo apt install -y python3-venv python3-picamera2 ent
python3 -m venv --system-site-packages ~/puffer && source ~/puffer/bin/activate
pip install "puffer-cipher[pi]"
PUFFER_SOURCE=picamera PUFFER_LED=gpio:17 PUFFER_LIVENESS=required puffer serve --host 0.0.0.0
```

`--system-site-packages` makes the distribution's `picamera2` visible inside
the virtual environment. With Docker use the `pi` profile in
`docker-compose.yml`, which passes `/dev/video0` and `/dev/gpiochip0` through.

## Placement checklist

- Camera fixed rigidly (vibration looks like global motion and is filtered out).
- Whole tank in view, LED in a corner, no window reflections directly on the lens.
- The LED region and a small border are excluded from sensor sampling: the LED
  pattern is our own challenge, never counted as entropy.
- Keep the aquarium's day/night light on a timer; the dashboard shows the
  entropy shift between regimes.
