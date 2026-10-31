# Training a YOLO detector for your own fish

The default detector (`detector = "motion"`) needs no training: a
background model plus blob and trajectory tests (see `puffer.vision.detector`
and `puffer.vision.classify`). It is measured on the simulator in the README
(`scripts/eval_tracking.py`). A detector trained on **your** tank does
better where blob detection is weakest:

* fish that overlap or touch (a blob detector sees one blob; a trained
  detector sees two fish),
* fish that rest from the moment the camera starts (a motion detector has
  never seen them move, so the tracker will not credit them),
* clutter the motion pipeline was not designed for (a filter outlet that
  moves like a fish, reflections on the front glass),
* and, trained with **one class per fish**, it names the fish directly,
  including at night where colour re-identification is impossible
  (IR frames are grayscale).

The trained model plugs in with `detector = "yolo:<weights.pt>"`; the runtime
then enables `FishReID(trust_labels=True)`, so a detection labelled
`"Nemo"` *is* Nemo. Everything downstream (ByteTrack, the track classifier,
the fish entropy channel) is unchanged.

`ultralytics` (and PyTorch) is an optional dependency, only needed for the
training step: `pip install 'puffer-cipher[yolo]'`. Exporting datasets works
in the base install.

## The workflow

```
record ──► pseudo-label ──► correct (CVAT / Label Studio) ──► train ──► run
  │             │                        │                       │         │
tank.mp4   FishTracker boxes     fix misses, clutter,      best.pt   detector=
           (fish tracks only)    identities                          "yolo:best.pt"
```

### 1. Record

Record the tank the way the camera will see it in production: same
position, same resolution, the lights you actually use. Cover what varies:
day and night (IR), lights switching, the bubbler on and off, feeding time,
fish resting. A few short clips spread over a day beat one long clip.

```bash
puffer simulate -o demo.mp4      # (the simulator equivalent, for a dry run)
# from a real camera, e.g. with ffmpeg or the camera app, 640x480 or larger
```

### 2. Pseudo-label

Let the tracker do the tedious part:

```bash
python scripts/train_yolo.py pseudo-label --source tank.mp4 --out data/yolo-tank \
    --every 10 --warmup 150            # add --per-fish for one class per fish
```

Every frame is tracked (tracking needs consecutive frames); every 10th frame
after a 5-second warm-up is exported. Only tracks the classifier accepted
as fish and that had a real measurement in that frame are labelled, with
the detector's box. Frames without any fish are exported with empty label
files: these negatives teach the detector that your plants are not fish.
The dataset is split into train/val by **time blocks**, not frame by frame,
so validation images are not near-duplicates of training images.

`provenance.json` in the output records that these are pseudo-labels.
Treat them as a draft. With `--per-fish`, labels are the names the
re-identification gave the fish (unidentified fish are labelled `fish`);
check that each name sticks to the same fish.

### 3. Correct

The output folder is in the Ultralytics YOLO layout (`dataset.yaml`,
`images/{train,val}`, `labels/{train,val}`), which annotation tools import:

* **CVAT**: create a task with the images, then *Actions → Upload
  annotations → "Ultralytics YOLO Detection 1.0"* with the zipped folder.
  Correct, then *Export task dataset* in the same format.
* **Label Studio**: convert the labels with Label Studio's converter
  (`label-studio-converter import yolo ...`, see its documentation for the
  flags of your version), correct, and export as *YOLO*.

What to fix, in order of importance:

1. **Boxes on things that are not fish** (a leaf, a reflection, a snail).
   Delete them; these are the errors that would end up credited as fish
   entropy if the detector learned them.
2. **Missed fish**: draw them, especially fish that overlap and fish at
   rest.
3. **Identities** (per-fish datasets): one class per individual, the same
   name everywhere. Class names become the names in the dashboard and in
   entropy receipts.
4. Box tightness: label what is visible, fins included.

### 4. Train

```bash
pip install 'puffer-cipher[yolo]'
python scripts/train_yolo.py train --data data/yolo-tank/dataset.yaml \
    --model yolo11n.pt --epochs 80 --imgsz 640
```

This fine-tunes a small pretrained model (`yolo11n.pt`; a Raspberry Pi 5
runs the nano model at a usable rate, a desktop GPU anything). Hue
augmentation is kept mild on purpose: colour is what tells individual fish
apart. Watch the validation mAP and look at the validation predictions
Ultralytics writes to `runs/puffer/fish/`: if a plant is detected as a
fish there, add more negative frames of that plant.

To test the pipeline without footage, export a simulator dataset with
**perfect** labels (one class per fish) and train on it:

```bash
python scripts/train_yolo.py simulate --out data/yolo-sim --frames 6000 --seed 7
python scripts/train_yolo.py train --data data/yolo-sim/dataset.yaml --epochs 30
```

Simulator labels are *amodal* (a fish hidden behind a plant is still
labelled where it is) and simulated fish are not your fish; a model trained
only on them is a smoke test, not a detector for a real tank.

### 5. Run

```bash
PUFFER_DETECTOR=yolo:runs/puffer/fish/weights/best.pt puffer serve
# or in puffer.toml:   [puffer]  detector = "yolo:runs/puffer/fish/weights/best.pt"
```

In Python:

```python
from puffer.vision import FishReID, FishTracker, YoloDetector

tracker = FishTracker(
    detector=YoloDetector("runs/puffer/fish/weights/best.pt", conf=0.25),
    reid=FishReID(trust_labels=True),
)
```

Keep `conf` low (0.25): ByteTrack uses low-confidence boxes to keep existing
tracks alive through occlusion; only confident boxes start tracks. The
track classifier still applies (a track becomes a fish once it has
travelled); for a detector you trust completely,
`FishTracker(classifier=TrackClassifier.permissive())` credits fish from
their first frame.

## Measuring the result

On the simulator, `scripts/eval_tracking.py` scores any tracker against
ground truth. On your tank, label a separate validation clip by hand (not
pseudo-labels: the tracker would be graded against itself) and compare the
motion detector with the trained one using
`puffer.vision.evaluate.evaluate_tracking` with a source that implements
`ground_truth()` from your labels.
