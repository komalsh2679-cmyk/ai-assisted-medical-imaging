# MedVision AI — Starter Repository

An educational research prototype for AI-assisted 3D CT segmentation.

## Project layout

```text
MedVision_AI_Starter/
├── interface/
│   └── app.py
├── checkpoints/
│   └── unet3d_lung_latest.pth   # copy your trained checkpoint here
├── results/
├── requirements.txt
└── README.md
```

## Setup in VS Code

1. Open this folder in VS Code.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Copy your existing `unet3d_lung_latest.pth` checkpoint from Google Drive into:

   `checkpoints/unet3d_lung_latest.pth`

5. Run the app from the project root:

   ```bash
   python interface/app.py
   ```

6. Open the local Gradio URL printed in the terminal.

## Important compatibility note

The model architecture and preprocessing in `interface/app.py` are based on
the current project details: MONAI 1.6.1, a 3D U-Net with one input channel,
two output classes, channels `(16, 32, 64, 128)`, strides `(2, 2, 2)`,
two residual units, and preprocessing to 1.5 mm spacing with intensity
scaling and foreground cropping.

Before final submission, compare these settings with the exact training
notebook. Any mismatch must be corrected. The current validation mean Dice
reported for 13 cases is approximately 0.0206. this Does not claim clinical accuracy.

## Notes

- This app expects a 3D NIfTI CT scan (`.nii` or `.nii.gz`).
- CPU inference may be slow. A CUDA-enabled PyTorch environment is needed
  for GPU inference.
- This is an educational/research prototype, not a medical device or
  diagnostic tool.
