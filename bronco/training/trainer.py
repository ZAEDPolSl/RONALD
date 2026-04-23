import torch
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from monai.data import set_track_meta
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
import os
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_function,
        device,
        output_dir,
        patience=50,
        min_delta=1e-4,
        amp=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.output_dir = output_dir
        self.patience = patience
        self.min_delta = min_delta
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
        os.makedirs(output_dir, exist_ok=True)

    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs,
        val_interval=1,
        plot_interval=10,
        checkpoint_interval=50,
        sw_batch_size=4,
        roi_size=(96, 96, 96),
    ):
        best_metric = -1
        best_metric_epoch = -1
        epochs_no_improve = 0
        early_stop = False
        train_losses = []
        val_metrics = []
        val_per_class_metrics = []
        learning_rates = []
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            step = 0
            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False
            )
            for batch_data in train_pbar:
                step += 1
                inputs, labels = batch_data["image"].to(self.device), batch_data[
                    "label"
                ].to(self.device)
                self.optimizer.zero_grad()
                if self.amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            epoch_loss /= step
            train_losses.append(epoch_loss)
            learning_rates.append(self.optimizer.param_groups[0]["lr"])
            # Log per-epoch metrics
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            print(f"Epoch {epoch+1}/{max_epochs} - Train loss: {epoch_loss:.4f}")
            if (epoch + 1) % val_interval == 0:
                val_metric, per_class_dice, val_vis, val_loss = self.validate(
                    val_loader,
                    sw_batch_size,
                    roi_size,
                    visualize=(epoch + 1) % 10 == 0,
                    show_progress=True,
                )
                val_metrics.append(val_metric)
                val_per_class_metrics.append(per_class_dice)
                self.writer.add_scalar("Dice/val", val_metric, epoch)
                # Log validation loss
                if val_loss is not None:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                for i, d in enumerate(per_class_dice):
                    self.writer.add_scalar(f"Dice/val_class_{i}", d, epoch)
                print(
                    f"Validation Dice: {val_metric:.4f} | Per-class: {per_class_dice}"
                )
                # Every 10 epochs, log image visualizations
                if val_vis is not None:
                    for idx, vis_dict in enumerate(val_vis):
                        for plane, (img, gt, pred) in vis_dict.items():
                            self.writer.add_image(
                                f"Val/{plane}_Image_{idx}", img, epoch, dataformats="HW"
                            )
                            self.writer.add_image(
                                f"Val/{plane}_GT_{idx}", gt, epoch, dataformats="HW"
                            )
                            self.writer.add_image(
                                f"Val/{plane}_Pred_{idx}", pred, epoch, dataformats="HW"
                            )
                if val_metric > best_metric + self.min_delta:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    epochs_no_improve = 0
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output_dir, "best_model.pth"),
                    )
                else:
                    epochs_no_improve += val_interval
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        early_stop = True
                        break
            self.scheduler.step()
            if (epoch + 1) % checkpoint_interval == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                )
        self.writer.close()
        print(
            f"Training complete. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}"
        )
        return best_metric, best_metric_epoch

    def validate(
        self, loader, sw_batch_size, roi_size, visualize=False, show_progress=False
    ):
        self.model.eval()
        dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        post_pred = AsDiscrete(argmax=True, to_onehot=4)
        post_label = AsDiscrete(to_onehot=4)
        vis_samples = []
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            val_iter = (
                tqdm(loader, desc="Validation", leave=False)
                if show_progress
                else loader
            )
            for val_data in val_iter:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, self.model
                )
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels)
                val_output_convert = [post_pred(x) for x in val_outputs_list]
                val_labels_convert = [post_label(x) for x in val_labels_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                if show_progress:
                    # Show running mean dice for this batch if available
                    try:
                        agg = dice_metric.aggregate()
                        if agg is not None:
                            batch_dice = agg.cpu().numpy().mean().item()
                            val_iter.set_postfix({"mean_dice": f"{batch_dice:.4f}"})
                    except Exception:
                        # If aggregation is not ready yet, skip updating the progress bar
                        pass
                # Visualization: only for the first batch, only if requested
                if visualize and len(vis_samples) == 0:
                    for i in range(min(5, len(val_outputs_list))):
                        img_np = val_inputs[i].detach().cpu().numpy()
                        gt_np = val_labels[i].detach().cpu().numpy()
                        pred_np = val_outputs[i].detach().cpu().numpy()
                        # Assume (C, D, H, W)
                        if img_np.ndim == 4:
                            c, d, h, w = img_np.shape
                            # Axial (horizontal): middle slice along D
                            mid_d = d // 2
                            axial_img = img_np[0, mid_d, :, :]
                            axial_gt = (
                                gt_np[0, mid_d, :, :]
                                if gt_np.shape[0] == 1
                                else gt_np[mid_d, :, :]
                            )
                            axial_pred = (
                                np.argmax(pred_np[:, mid_d, :, :], axis=0)
                                if pred_np.shape[0] > 1
                                else pred_np[0, mid_d, :, :]
                            )
                            # Coronal: middle slice along H
                            mid_h = h // 2
                            coronal_img = img_np[0, :, mid_h, :]
                            coronal_gt = (
                                gt_np[0, :, mid_h, :]
                                if gt_np.shape[0] == 1
                                else gt_np[:, mid_h, :]
                            )
                            coronal_pred = (
                                np.argmax(pred_np[:, :, mid_h, :], axis=0)
                                if pred_np.shape[0] > 1
                                else pred_np[0, :, mid_h, :]
                            )
                            # Sagittal: middle slice along W
                            mid_w = w // 2
                            sagittal_img = img_np[0, :, :, mid_w]
                            sagittal_gt = (
                                gt_np[0, :, :, mid_w]
                                if gt_np.shape[0] == 1
                                else gt_np[:, :, mid_w]
                            )
                            sagittal_pred = (
                                np.argmax(pred_np[:, :, :, mid_w], axis=0)
                                if pred_np.shape[0] > 1
                                else pred_np[0, :, :, mid_w]
                            )

                            # Normalize for visualization
                            def norm(x):
                                return (x - x.min()) / (np.ptp(x) + 1e-8)

                            vis_dict = {
                                "Axial": (
                                    norm(axial_img),
                                    (axial_gt > 0).astype(np.float32),
                                    (axial_pred > 0).astype(np.float32),
                                ),
                                "Coronal": (
                                    norm(coronal_img),
                                    (coronal_gt > 0).astype(np.float32),
                                    (coronal_pred > 0).astype(np.float32),
                                ),
                                "Sagittal": (
                                    norm(sagittal_img),
                                    (sagittal_gt > 0).astype(np.float32),
                                    (sagittal_pred > 0).astype(np.float32),
                                ),
                            }
                        else:
                            img_slice = np.squeeze(img_np)
                            gt_slice = np.squeeze(gt_np)
                            pred_slice = np.squeeze(pred_np)
                            vis_dict = {
                                "Axial": (
                                    (img_slice - img_slice.min())
                                    / (img_slice.ptp() + 1e-8),
                                    (gt_slice > 0).astype(np.float32),
                                    (pred_slice > 0).astype(np.float32),
                                )
                            }
                        vis_samples.append(vis_dict)
        dice_scores = dice_metric.aggregate().cpu().numpy()
        mean_dice = dice_scores.mean().item()
        dice_metric.reset()
        mean_val_loss = None
        if val_steps > 0:
            mean_val_loss = val_loss_sum / val_steps
        return (
            mean_dice,
            dice_scores,
            vis_samples if visualize and vis_samples else None,
            mean_val_loss,
        )
