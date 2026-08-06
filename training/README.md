# KiPA22 renal-vessel machine learning

This folder contains the reproducible training and export path for the application's real
renal artery/vein model. It intentionally does not contain KiPA22 data or trained weights.

## No-training-data route: published BA-Net checkpoint

If you do not have authorized KiPA22 training data, open
[`notebooks/ba_net_single_fold_inference.ipynb`](../notebooks/ba_net_single_fold_inference.ipynb)
in Google Colab. It runs one fold from the authors' public Zenodo checkpoint and produces a
small prediction ZIP for the Streamlit uploader. It reads the remote ZIP index and retrieves
only the requested BA-Net members (roughly 355 MB compressed), rather than downloading the
full 3.7 GB BA-Net + ResUNet ensemble.

This is pretrained inference, not retraining or validation. The output package is accepted
only when its case ID, exact CT SHA-256, canonical shape, affine, label set, and research-only
provenance match the selected case. BA-Net was trained on cropped KiPA22 CTA, so results on
KiTS23 and other contrast phases remain an unvalidated domain transfer.

## Why KiPA22

KiPA22 is CTA data with expert labels for renal vein, kidney, renal artery, and kidney tumor.
KiTS23 does not label vessels and is not used as vessel ground truth. KiPA22 requires each
participant to join the challenge, complete its data form, and accept its research-only terms.
The project must not copy the dataset into GitHub or redistribute its download links.

## Model

The training program uses a compact residual 3D U-Net with:

- RAS orientation and 1 mm isotropic preprocessing;
- vessel-balanced 96³ patch sampling;
- class-weighted Dice and cross-entropy segmentation loss;
- centerline-aware soft-clDice loss for arterial/venous continuity;
- deterministic held-out folds;
- sliding-window full-volume validation;
- separate artery and vein Dice release gates;
- fixed-patch ONNX export for portable CPU inference.

The default validation gates are artery Dice >= 0.70 and vein Dice >= 0.65. Failing a gate
still writes the experimental artifact, but marks it unvalidated; the Streamlit application
will refuse to load it.

## Free GPU training

Google AI Studio is for Gemini model development and is not an arbitrary PyTorch training
host. Use a free Google Colab or Kaggle GPU session instead. Keep the authorized KiPA22 folder
in your private Drive/dataset storage rather than a public notebook or repository.

In a GPU notebook terminal:

```bash
git clone https://github.com/ahmederaky2-cmyk/renal-ct-explorer.git
cd renal-ct-explorer
python -m pip install -r requirements-training.txt
python training/train_kipa22.py \
  --data /private/path/to/KiPA22/dataset \
  --output /private/path/to/kipa22_vessel_unet_v1 \
  --fold 0 \
  --epochs 250
```

The output folder contains:

```text
kipa22_vessel_unet_v1/
├── best_model.pt       # training checkpoint; not needed by Streamlit
├── model.onnx          # portable inference artifact
└── manifest.json       # data/model provenance, metrics, gate result, SHA-256
```

For a stronger result, train all five folds and compare them on a locked test cohort before
choosing an artifact. Do not select a model using the challenge open test labels.

## Serve the validated model

Copy only `model.onnx` and `manifest.json` into a private server directory and set:

```bash
export RENAL_VESSEL_MODEL_DIR=/absolute/path/to/kipa22_vessel_unet_v1
streamlit run app.py
```

At startup the application verifies that the manifest says `validated: true` and that the
ONNX file matches its recorded SHA-256. The **Interactive 3D** tab then offers
**KiPA22 3D U-Net (ML)** and renders artery and vein as separate surfaces. If no validated
model is installed, the app says so and exposes only the explicitly non-ML filter fallback.

For free Streamlit hosting, upload `model.onnx` and `manifest.json` as private or public
HTTPS release assets that comply with the KiPA22 agreement, then configure Streamlit secrets
or environment variables:

```bash
RENAL_VESSEL_MODEL_URL=https://host.example/model.onnx
RENAL_VESSEL_MANIFEST_URL=https://host.example/manifest.json
```

The server downloads at most 200 MB into temporary storage and checks the manifest SHA-256
before loading ONNX. The exported compact model is normally around 20 MB; do not deploy a
multi-gigabyte ensemble on Streamlit Community Cloud.

This remains research software. KiPA22 is cropped CTA; applying the model to routine venous,
non-contrast, or KiTS23 CT is a domain shift requiring external validation.
