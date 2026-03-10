import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os


def log_confidence_maps_mlflow(conf_gt, conf_pred, conf_spvs, epoch, spvs_ratio, trainer):
    """
    Creates a figure with 3 subplots (ground truth, supervision, prediction)
    and logs it to MLflow.
    """
    # Convert tensors to NumPy arrays.
    gt = conf_gt[0].cpu().numpy()
    pred = conf_pred[0].cpu().numpy()
    spvs = conf_spvs[0].cpu().numpy()

    # Create a figure with 3 subplots.
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))

    fig.suptitle(f"Confidence Maps - Epoch: {epoch}, Supervision ratio: {spvs_ratio}", fontsize=16)

    axs[0].imshow(pred, cmap='viridis')
    axs[0].set_title('conf_pred')


    axs[2].imshow(gt, cmap='viridis')
    axs[2].set_title('conf_gt')


    axs[1].imshow(spvs, cmap='viridis')
    axs[1].set_title('conf_spvs')


    plt.tight_layout()
    mlflow_logger = trainer.logger
    mlflow_logger.experiment.log_figure(mlflow_logger.run_id, fig, f"confidence_maps_epoch_{epoch}.png")
    plt.close(fig)


def log_confidence_maps2(conf_gt, conf_pred, epoch, type, trainer):
    # Convert tensors to NumPy arrays.
    gt = conf_gt[0].detach().cpu().numpy()
    pred = conf_pred[0].detach().cpu().numpy()

    # Determine the coarse min/max across both images (so both subplots share the same scale).
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    # Create a figure with 2 subplots, using constrained_layout for better spacing.
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    fig.suptitle(f"Confidence Maps - Epoch: {epoch}", fontsize=16)

    # Plot Predicted Confidence Map
    im_pred = axs[0].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('Prediction (conf_pred)')

    # Plot Ground Truth Confidence Map
    im_gt = axs[1].imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('Ground Truth (conf_gt)')

    # Create ONE colorbar for both images.
    # By passing ax=axs (the list of axes), the colorbar will match the combined height.
    cbar = fig.colorbar(im_gt, ax=axs, orientation='vertical')
    cbar.set_label("Confidence")

    # Log figure to MLflow
    mlflow_logger = trainer.logger
    mlflow_logger.experiment.log_figure(
        mlflow_logger.run_id, fig, f"{type} confidence_maps_epoch_{epoch}.png"
    )
    plt.close(fig)
def log_confidence_maps2_plotly(conf_gt, conf_pred, epoch, type, trainer):
    # Convert tensors to NumPy arrays.
    gt = conf_gt[0].cpu().detach().numpy()
    pred = conf_pred[0].cpu().detach().numpy()

    # Determine the coarse min/max across both images.
    vmin = float(min(gt.min(), pred.min()))
    vmax = float(max(gt.max(), pred.max()))

    # Create a Plotly figure with 2 subplots.
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Prediction (conf_pred)", "Ground Truth (conf_gt)"))

    # Create interactive heatmaps with hover info.
    trace_pred = go.Heatmap(
        z=pred,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title="Confidence", len=0.75),
        hovertemplate="Value: %{z}<extra></extra>"
    )

    trace_gt = go.Heatmap(
        z=gt,
        colorscale='Viridis',
        zmin=vmin,
        zmax=vmax,
        showscale=False,  # Only show one colorbar.
        hovertemplate="Value: %{z}<extra></extra>"
    )

    fig.add_trace(trace_pred, row=1, col=1)
    fig.add_trace(trace_gt, row=1, col=2)

    # Update layout.
    fig.update_layout(title_text=f"Confidence Maps - Epoch: {epoch}", height=600, width=1000)

    # Save the interactive figure as an HTML file.
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        html_filename = tmpfile.name
    fig.write_html(html_filename)

    # Log the HTML artifact to MLflow so you can open it in a browser and hover to see values.
    mlflow_logger = trainer.logger
    mlflow_logger.experiment.log_artifact(
        mlflow_logger.run_id, html_filename, artifact_path=f"{type}_confidence_maps_epoch_{epoch}.html"
    )

    # Clean up the temporary file.
    os.remove(html_filename)

def infer_from_similarity(config, similarity):
    coarse_factor = config.data.coarse_factor
    image_shape = config.data.image_shape
    w = image_shape[0] // coarse_factor
    i_ids, j_ids = similarity.nonzero()
    mask0, mask1 = np.zeros( w * w, dtype=np.uint8), np.zeros(w * w, dtype=np.uint8)
    mask0[i_ids] = 1
    mask1[j_ids] = 1

    mask0 = mask0.reshape(w, w)
    mask1 = mask1.reshape(w, w)

    patch_size = (coarse_factor, coarse_factor)

    mask0 = np.kron(mask0, np.ones(patch_size, dtype=mask0.dtype))
    mask1 = np.kron(mask1, np.ones(patch_size, dtype=mask1.dtype))
    return mask0, mask1


def retrieve_masks(conf_c_pred, conf_c_gt, config):
    conf_thresh = config.model.conf_thresh
    mask0_gt, mask1_gt  = infer_from_similarity(config, conf_c_gt)
    mask0_pred, mask1_pred = infer_from_similarity(config, conf_c_pred>conf_thresh)
    return mask0_gt, mask1_gt, mask0_pred, mask1_pred


def log_coarse_pred(data, trainer, epoch, config):
    image0 = data['image0'][0].detach().cpu().numpy().squeeze(0)
    image1 = data['image1'][0].detach().cpu().numpy().squeeze(0)
    conf_c_pred = data['conf_c_pred'][0].detach().cpu().numpy()

    image_shape = config.data.image_shape
    h, w = image_shape
    conf_c_gt = data['conf_c_gt'][0].detach().cpu().numpy()
    mask0_gt, mask1_gt, mask0_pred, mask1_pred = retrieve_masks(conf_c_pred, conf_c_gt, config)

    fig, axs = plt.subplots(1, 2, figsize=(8, 10))
    axs[0].imshow(image0, cmap='gray')
    axs[0].imshow(mask0_gt, cmap='Greens',
                  alpha=0.3,                        # mask visibility
                  extent=[0, h, w, 0],
                  interpolation='nearest')

    # Overlay the predicted mask in blue
    axs[0].imshow(mask0_pred, cmap='Blues',
                  alpha=0.3,
                  extent=[0, h, w, 0],
                  interpolation='nearest')

    axs[0].set_title('Image 0')

    axs[1].imshow(image1, cmap='gray')
    axs[1].imshow(mask1_gt, cmap='Greens',
                  alpha=0.3,
                  extent=[0, h, w, 0],
                  interpolation='nearest')

    axs[1].imshow(mask1_pred, cmap='Blues',
                  alpha=0.3,
                  extent=[0, h,w, 0],
                  interpolation='nearest')

    axs[1].set_title('Image 1')

    mlflow_logger = trainer.logger
    mlflow_logger.experiment.log_figure(
        run_id=mlflow_logger.run_id, figure=fig, artifact_file=f"coarse_pred_epoch_{epoch}.png",
    )
    plt.close(fig)


def log_train_val_losses(trainer, train_losses, val_losses):
    """
    Plot training and validation losses.

    Args:
        trainer (Trainer): PyTorch Lightning trainer instance.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        epoch (int): Current epoch number.
        out_dir (str): Directory to save the plot.
    """
    # 1) build the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    if val_losses:
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    ax.set_title(f"Training and Validation Losses")

    # 2) save to file
    fname = f"loss_graphs/{trainer.current_epoch}.png"


    # 4) log as artifact if supported
    logger = trainer.logger
    logger.experiment.log_figure(logger.run_id, fig, fname)
    plt.close(fig)




def log_f1_scores(trainer, train_f1, val_f1):
    """
    Plot training and validation losses.

    Args:
        trainer (Trainer): PyTorch Lightning trainer instance.
        train_f1 (list): List of training f1 scores.
        val_f1 (list): List of validation f1 scores.

    """
    # 1) build the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_f1) + 1), train_f1, label="Train F1 Score")
    if val_f1:
        ax.plot(range(1, len(val_f1) + 1), val_f1, label="Val F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid()
    ax.set_title(f"Training and Validation Losses")

    # 2) save to file
    fname = "train_val_f1.png"


    # 4) log as artifact if supported
    logger = trainer.logger
    logger.experiment.log_figure(logger.run_id, fig, fname)
    plt.close(fig)