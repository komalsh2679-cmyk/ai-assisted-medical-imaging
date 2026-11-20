from pathlib import Path
import uuid
import warnings

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
)

warnings.filterwarnings("ignore")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
UNET_CHECKPOINT_PATH = CHECKPOINT_DIR / "unet3d_lung_latest.pth"
DENSENET_CHECKPOINT_PATH = CHECKPOINT_DIR / "densenet121_epoch1.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "interface_outputs"
XRAY_OUTPUT_DIR = PROJECT_ROOT / "results" / "xray_gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
XRAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# CT SCAN: MONAI 3D U-NET
# ---------------------------------------------------------------------
def build_unet_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)

    if not UNET_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {UNET_CHECKPOINT_PATH}. "
            "Place unet3d_lung_latest.pth in the checkpoints folder."
        )

    checkpoint = torch.load(
        UNET_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        state = checkpoint["model"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError("Unrecognized U-Net checkpoint structure.")

    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


CT_PREPROCESS = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
    CropForegroundd(keys=["image"], source_key="image"),
    EnsureTyped(keys=["image"]),
])

UNET_MODEL = None
UNET_MODEL_ERROR = None
try:
    UNET_MODEL = build_unet_model()
    print(f"3D U-Net loaded on {DEVICE}.")
except Exception as exc:
    UNET_MODEL_ERROR = str(exc)
    print("3D U-Net has not loaded:", UNET_MODEL_ERROR)


def analyze_ct_scan(uploaded_file):
    if uploaded_file is None:
        raise gr.Error("Upload a CT scan in NIfTI format first.")

    if UNET_MODEL is None:
        raise gr.Error(
            "The 3D U-Net could not be loaded. Check "
            f"checkpoints/unet3d_lung_latest.pth. Details: {UNET_MODEL_ERROR}"
        )

    input_path = Path(uploaded_file)
    if not (
        input_path.name.lower().endswith(".nii")
        or input_path.name.lower().endswith(".nii.gz")
    ):
        raise gr.Error("Only .nii and .nii.gz files are supported.")

    try:
        sample = CT_PREPROCESS({"image": str(input_path)})
        image = sample["image"].unsqueeze(0)

        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=UNET_MODEL,
                overlap=0.25,
                sw_device=DEVICE,
                device="cpu",
            )

        prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        image_volume = image[0, 0].cpu().numpy()
        z = image_volume.shape[2] // 2

        ct_slice = image_volume[:, :, z]
        mask_slice = prediction[:, :, z]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("Preprocessed CT")
        axes[1].imshow(mask_slice.T, cmap="gray", origin="lower")
        axes[1].set_title("Predicted mask")
        axes[2].imshow(ct_slice.T, cmap="gray", origin="lower")
        if np.any(mask_slice):
            axes[2].contour(
                mask_slice.T,
                levels=[0.5],
                colors=["red"],
                linewidths=1,
                origin="lower",
            )
        else:
            axes[2].text(
                0.5, 0.5, "No foreground predicted",
                color="yellow", ha="center", va="center",
                transform=axes[2].transAxes,
            )
        axes[2].set_title("CT + predicted mask")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()

        token = uuid.uuid4().hex[:10]
        preview_path = OUTPUT_DIR / f"{token}_ct_preview.png"
        mask_path = OUTPUT_DIR / f"{token}_prediction.nii.gz"
        fig.savefig(preview_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        affine = sample["image"].affine.cpu().numpy()
        nib.save(nib.Nifti1Image(prediction, affine), str(mask_path))

        foreground_percent = float(np.mean(prediction > 0) * 100)
        summary = (
            f"Input file: {input_path.name}\n"
            f"Device: {DEVICE}\n"
            f"Output volume shape: {prediction.shape}\n"
            f"Predicted foreground: {foreground_percent:.4f}%"
        )
        return str(preview_path), str(mask_path), summary

    except Exception as exc:
        raise gr.Error(f"CT analysis failed: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------
# CHEST X-RAY: DENSENET121 + GRAD-CAM
# ---------------------------------------------------------------------
XRAY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

XRAY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

DENSENET_MODEL = None
DENSENET_MODEL_ERROR = None


def build_densenet_model():
    if not DENSENET_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {DENSENET_CHECKPOINT_PATH}. "
            "Place densenet121_epoch1.pth in the checkpoints folder."
        )

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(XRAY_LABELS))

    checkpoint = torch.load(
        DENSENET_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise RuntimeError("Unrecognized DenseNet checkpoint structure.")

    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


try:
    DENSENET_MODEL = build_densenet_model()
    print(f"DenseNet121 loaded on {DEVICE}.")
except Exception as exc:
    DENSENET_MODEL_ERROR = str(exc)
    print("DenseNet121 has not loaded:", DENSENET_MODEL_ERROR)


def analyze_chest_xray(uploaded_file):
    if uploaded_file is None:
        raise gr.Error("Upload a chest X-ray image first.")

    if DENSENET_MODEL is None:
        raise gr.Error(
            "DenseNet121 could not be loaded. Check "
            f"checkpoints/densenet121_epoch1.pth. Details: {DENSENET_MODEL_ERROR}"
        )

    input_path = Path(uploaded_file)
    try:
        original_image = Image.open(input_path).convert("L")
        image_tensor = XRAY_TRANSFORM(original_image).unsqueeze(0).to(DEVICE)

        activations = {}
        gradients = {}
        target_layer = DENSENET_MODEL.features.denseblock4

        forward_handle = target_layer.register_forward_hook(
            lambda _module, _inputs, output: activations.__setitem__("value", output)
        )

        def save_gradient(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0]

        backward_handle = target_layer.register_full_backward_hook(save_gradient)

        try:
            DENSENET_MODEL.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = DENSENET_MODEL(image_tensor)
                probabilities = torch.sigmoid(logits)[0]
                predicted_index = int(torch.argmax(probabilities).item())
                logits[0, predicted_index].backward()

                feature_maps = activations["value"]
                grads = gradients["value"]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(
                    cam,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                cam = cam.detach().cpu().numpy()
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()

                probability_values = probabilities.detach().cpu().numpy()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        # Create a readable grayscale original with a Grad-CAM overlay.
        display_image = original_image.resize((224, 224))
        display_array = np.asarray(display_image, dtype=np.float32) / 255.0

        fig, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(display_array, cmap="gray")
        axis.imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
        axis.set_title(f"Grad-CAM: {XRAY_LABELS[predicted_index]}")
        axis.axis("off")
        fig.tight_layout()

        token = uuid.uuid4().hex[:10]
        heatmap_path = XRAY_OUTPUT_DIR / f"{token}_gradcam.png"
        fig.savefig(heatmap_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        ranked = sorted(
            zip(XRAY_LABELS, probability_values),
            key=lambda item: item[1],
            reverse=True,
        )
        probability_text = (
            f"Highest-scoring label: {XRAY_LABELS[predicted_index]}\n\n"
            "Model scores (highest to lowest):\n"
            + "\n".join(f"{label}: {score * 100:.2f}%" for label, score in ranked)
        )
        return str(heatmap_path), probability_text

    except Exception as exc:
        raise gr.Error(
            f"Chest X-ray analysis failed: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# GRADIO INTERFACE
# ---------------------------------------------------------------------
with gr.Blocks(title="MedVision AI") as demo:
    gr.Markdown(
        """
        # MedVision AI
        ### AI-Assisted Medical Imaging
        """
    )

    with gr.Tab("CT Scan Analysis"):
        gr.Markdown("Upload a 3D CT scan in NIfTI format to generate a segmentation preview and mask.")
        with gr.Row():
            with gr.Column(scale=1):
                ct_input = gr.File(
                    label="Upload CT scan (.nii / .nii.gz)",
                    file_types=[".nii", ".nii.gz"],
                    type="filepath",
                )
                ct_run_button = gr.Button("Analyze CT Scan", variant="primary")
                ct_result_text = gr.Textbox(
                    label="Run information",
                    lines=5,
                    interactive=False,
                )
            with gr.Column(scale=2):
                ct_preview = gr.Image(label="CT and segmentation preview", type="filepath")
                ct_download = gr.File(label="Download predicted segmentation")

        ct_run_button.click(
            fn=analyze_ct_scan,
            inputs=ct_input,
            outputs=[ct_preview, ct_download, ct_result_text],
        )

    with gr.Tab("Chest X-ray + Heatmap"):
        gr.Markdown("Upload a chest X-ray image to view model scores and its Grad-CAM heatmap.")
        with gr.Row():
            with gr.Column(scale=1):
                xray_input = gr.File(
                    label="Upload chest X-ray (PNG / JPG)",
                    file_types=[".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                xray_run_button = gr.Button("Analyze X-ray", variant="primary")
                xray_scores = gr.Textbox(
                    label="Model scores",
                    lines=17,
                    interactive=False,
                )
            with gr.Column(scale=1):
                xray_heatmap = gr.Image(
                    label="Grad-CAM heatmap",
                    type="filepath",
                )

        xray_run_button.click(
            fn=analyze_chest_xray,
            inputs=xray_input,
            outputs=[xray_heatmap, xray_scores],
        )

    gr.Markdown(
        """
        ---
        **Models:** MONAI 3D U-Net for CT segmentation · DenseNet121 for chest X-ray classification and Grad-CAM
        """
    )


if __name__ == "__main__":
    demo.launch()
