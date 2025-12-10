# Gestura AI

Indonesian Sign Language (BISINDO) classifier built with TensorFlow. The project trains on hand-sign images and provides a simple webcam tester for live predictions.

## Requirements

- Python 3.9+ (tested with TensorFlow)
- Virtual environment recommended
- For webcam test: a working camera and OpenCV support

## Setup

- Linux / macOS
  ```bash
  # from the repo root
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Windows (PowerShell)
  ```powershell
  # from the repo root
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

## Data

Dataset comes from Kaggle: Indonesian Sign Language - BISINDO. See [`data/README.md`](/data/README.md) for download instructions and folder layout. After downloading, you should have `data/images/{train,val}` and `data/labels/{train,val}`.

## Train

Run from the project root with the virtual environment activated:

```bash
python -m gestura_ai.train
```

Force CPU (useful on laptops without a compatible GPU):

```bash
GESTURA_FORCE_CPU=1 python -m gestura_ai.train
```

Models and label metadata are saved under `models/` (`gestura_bisindo_best.h5`, `gestura_bisindo_last.h5`, `labels.json`).

## Webcam Test

After training (models present in `models/`), run the live classifier:

```bash
python -m gestura_ai.webcam_test
```

Press `q` to quit the window.

## License

Licensed under the GNU General Public License v3.0. See [`LICENSE`](/LICENSE) for details.
