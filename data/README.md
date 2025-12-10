# Dataset Guide

Indonesian Sign Language (BISINDO) images and bounding boxes for 26 alphabet classes (A–Z), sourced from Kaggle.

## Source

- Kaggle dataset: [Indonesian Sign Language - BISINDO](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo/data)

## Layout

```
data/
  images/
    train|val/<class>/*.jpg
  labels/
    train|val/<class>/*.txt   # YOLO format: class x_center y_center width height (normalized)
```

## Download (if you need to re-fetch)

1. Install the Kaggle CLI: `pip install kaggle`
2. Add your `kaggle.json` API token to `~/.kaggle/` (or set `KAGGLE_USERNAME`/`KAGGLE_KEY`).
3. Run `kaggle datasets download -d agungmrf/indonesian-sign-language-bisindo -p data --unzip`
4. Ensure the extracted `images/` and `labels/` directories match the layout above.
