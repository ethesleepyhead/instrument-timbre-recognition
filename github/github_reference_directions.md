# GitHub Reference Directions For `audio_ml`

Updated: 2026-05-28

## Why this file exists

The current local project already supports:

- uploaded audio classification
- Flask-backed web demo
- probability output per instrument
- sliding-window timeline visualization
- cached feature extraction and separate training flow

The goal of this note is to collect nearby GitHub directions that can improve the project without losing its "learning + presentation" focus.

## Local project snapshot

Current local codebase highlights:

- `index.html`, `script.js`, `style.css`: front-end demo UI
- `timbre_api.py`: Flask app, overall prediction, timeline prediction
- `train_model.py`: training script with quick mode and cache
- `features.py`: shared feature extraction logic
- `dataset/`: training data
- `timbre_model.pkl`: trained model artifact

This means the project is already beyond a toy prototype. The most valuable outside references are not generic "audio ML" repos, but repos that help with:

1. multi-instrument recognition
2. event-roll / timeline visualization
3. reusable dataset and feature workflows
4. demo-friendly browser experiences

## High-value reference projects

### 1. RiddhiRex/DetectingMusicInstruments

URL:

- <https://github.com/RiddhiRex/DetectingMusicInstruments>

Why it matters:

- It is the closest public project to the current one in spirit.
- It uses the same IRMAS-style instrument recognition setup.
- The README explicitly lists the same 11 instrument counts seen in the local dataset.
- It combines feature extraction, classification, and a web UI.

What to learn from it:

- how others package an instrument-classification demo for presentation
- how to describe the dataset and model story in a project README
- how to discuss multiple datasets as an upgrade path

Best use here:

- compare your README and UI story against it
- borrow ideas for dataset explanation and demo framing
- do not copy the old Python 2.7 / Essentia-heavy stack directly

### 2. OdysseasKr/irmas-cnn

URL:

- <https://github.com/OdysseasKr/irmas-cnn>

Why it matters:

- It is directly about instrument recognition on IRMAS.
- It documents the IRMAS split clearly: solo train clips, multi-instrument test clips.
- It includes a preprocessing class and keeps feature generation explicit.

What to learn from it:

- dataset-aware preprocessing
- clear separation between feature generation and model training
- a path from hand-crafted features to mel-based learning

Best use here:

- keep your current classical ML baseline
- later add a second model path using mel spectrogram inputs
- use it as a bridge if you want to explain "traditional features vs CNN features"

### 3. cosmir/openmic-2018

URL:

- <https://github.com/cosmir/openmic-2018>

Why it matters:

- It is a strong reference for multiple instrument recognition.
- OpenMIC-2018 is built for instrument presence labels rather than only one dominant class.
- It includes ready-to-use features and tutorial material.

What to learn from it:

- how multi-label instrument recognition differs from your current single-label setup
- how to structure a dataset with class maps, partitions, and precomputed embeddings
- how to think about "instrument present / not present" instead of "only one answer"

Best use here:

- strongest future direction if you want your timeline to become more honest for mixed music
- useful if you later move from "top-2 approximate coexistence" to real multi-label detection

### 4. TUT-ARG/sed_vis

URL:

- <https://github.com/TUT-ARG/sed_vis>

Why it matters:

- It is not about instruments specifically, but about visualizing sound events.
- Its event-roll idea is very close to your timeline card.
- It includes both interactive viewing and video generation concepts.

What to learn from it:

- event roll as a visualization language
- pairing waveform/spectrogram with temporal labels
- presentation-friendly output, including demo videos

Best use here:

- improve your current timeline UI
- add optional spectrogram + event-roll dual view
- add "export screenshot/video" later if you want a stronger presentation artifact

### 5. sharathadavanne/sed-crnn

URL:

- <https://github.com/sharathadavanne/sed-crnn>

Why it matters:

- It shows the frame-wise detection mindset more clearly than clip-level classification repos.
- The model predicts class activity over time rather than one answer for one clip.
- It is the conceptual next step after your current sliding-window approximation.

What to learn from it:

- frame-level probability outputs
- thresholding probabilities into active/inactive events
- event metrics and temporal evaluation

Best use here:

- not as a first implementation target
- very useful as the design reference if you later want true temporal detection
- especially useful for upgrading the timeline from "window voting" to "per-frame activity"

### 6. opsengine/onehotchord

URL:

- <https://github.com/opsengine/onehotchord>

Why it matters:

- It is a music-analysis demo with a browser-facing interface.
- The project separates model pipeline from web demo and openly states limitations.
- It treats the web UI as part of the product, not as an afterthought.

What to learn from it:

- how to present an ML music tool honestly
- how to keep the demo lightweight and approachable
- how to explain limitations without weakening the project

Best use here:

- improve your demo framing and project communication
- useful reference for "browser-first presentation" design choices

## Recommended directions for this local project

## Direction A: Keep the current project as a strong classical-ML baseline

This is the best short-term path.

Why:

- it is already running
- it is easy to explain in a classroom or portfolio setting
- MFCC/chroma/spectral features + random forest is teachable

Suggested improvements:

- fix all text encoding / Chinese garbling in UI and README
- clean up timeline visuals
- improve result explanation and model info display
- add example audio files for easier demoing

## Direction B: Make the timeline more "event-roll like"

This is the best presentation upgrade.

Inspired by:

- `sed_vis`

Suggested improvements:

- optional spectrogram under waveform
- cleaner segment merging
- per-label legends
- better hover tooltips
- exportable image of the timeline

## Direction C: Move toward real mixed-instrument recognition

This is the best modeling upgrade if you want to support overlapping instruments more honestly.

Inspired by:

- `openmic-2018`
- `sed-crnn`

Suggested improvements:

- use a multi-label dataset or weak labels
- change output from single-label probabilities to multi-label activity scores
- evaluate with thresholded per-class activity over time

Important note:

- this is a larger modeling change and is not necessary for the current "shareable demo" goal

## Direction D: Offer two model tracks inside the same project

This is the best learning-oriented upgrade.

Inspired by:

- `irmas-cnn`

Suggested structure:

- Track 1: classical features + random forest
- Track 2: mel spectrogram + CNN

Why this is valuable:

- easy to compare interpretability vs accuracy
- helps explain evolution from feature engineering to representation learning
- strong for learning reports and presentations

## Practical next steps

Priority 1:

- fix UI/README encoding problems
- keep startup and demo flow simple
- polish the timeline interface

Priority 2:

- add an explicit "research directions" section to the main README
- add sample audio files for demo
- add screenshots/GIFs of the page

Priority 3:

- add optional spectrogram visualization
- optimize timeline feature extraction path
- improve segment merging rules

Priority 4:

- experiment with OpenMIC-style multi-label recognition
- experiment with a mel-spectrogram CNN branch

## Source links

- RiddhiRex/DetectingMusicInstruments: <https://github.com/RiddhiRex/DetectingMusicInstruments>
- OdysseasKr/irmas-cnn: <https://github.com/OdysseasKr/irmas-cnn>
- cosmir/openmic-2018: <https://github.com/cosmir/openmic-2018>
- TUT-ARG/sed_vis: <https://github.com/TUT-ARG/sed_vis>
- sharathadavanne/sed-crnn: <https://github.com/sharathadavanne/sed-crnn>
- opsengine/onehotchord: <https://github.com/opsengine/onehotchord>
