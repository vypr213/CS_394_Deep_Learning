import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    # lr: float = 1e-3,
    lr: float = 0.005,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    train_metric_computer = DetectionMetric()
    val_metric_computer = DetectionMetric()

    train_metrics = None
    val_metrics = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train",
                           shuffle=True,
                           batch_size=batch_size,
                           num_workers=2)

    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func_1 = torch.nn.CrossEntropyLoss()
    loss_func_2 = torch.nn.L1Loss() # Make it L2 loss for regression
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            batch = {k: v.to(device)
                        if isinstance(v, torch.Tensor)
                        else v for k, v in batch.items()}

            img        = batch["image"]
            seg_labels = batch["track"]
            depth      = batch["depth"]

            # TODO: implement training step
            pred_segs, pred_depth = model(img)

            # print(img.shape)
            # print(pred_segs.shape, pred_segs.argmax(dim=1).shape, seg_labels.shape)
            # print(pred_depth.shape, depth.shape)
            loss_1 = loss_func_1(pred_segs, seg_labels)
            loss_2 = loss_func_2(pred_depth, depth)

            # print(loss_1.shape, loss_2.shape, loss_1, loss_2)

            loss_val = loss_1 + loss_2

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            logger.add_scalar("train_loss", loss_val, global_step)

            train_metric_computer.add(pred_segs.argmax(dim=1), seg_labels,
                                                      pred_depth, depth)


            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                batch = {k: v.to(device)
                            if isinstance(v, torch.Tensor)
                            else v for k, v in batch.items()}

                img        = batch["image"]
                seg_labels = batch["track"]
                depth      = batch["depth"]


                # TODO: compute validation accuracy
                val_segs, val_depth = model.predict(img)

                val_metric_computer.add(val_segs, seg_labels,
                                                      val_depth, depth)

                # raise NotImplementedError("Validation accuracy not implemented")

        # log average train and val accuracy to tensorboard
        # epoch_train_acc = torch.cat(metrics["train_acc"]).mean()
        # epoch_val_acc = torch.cat(metrics["val_acc"]).mean()

        # logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        # logger.add_scalar("val_accuracy",   epoch_val_acc,   global_step)
        # raise NotImplementedError("Logging not implemented")

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            train_metrics = train_metric_computer.compute()
            val_metrics = val_metric_computer.compute()
            train_acc          = train_metrics["accuracy"]
            train_iou          = train_metrics["iou"]
            train_depth_err    = train_metrics["abs_depth_error"]
            train_tp_depth_err = train_metrics["tp_depth_error"]

            val_acc          = val_metrics["accuracy"]
            val_iou          = val_metrics["iou"]
            val_depth_err    = val_metrics["abs_depth_error"]
            val_tp_depth_err = val_metrics["tp_depth_error"]

            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_acc:.4f} "
                f"train_iou={train_iou:.4f} "
                f"train_depth_err={train_depth_err:.4f} "
                f"train_tp_depth_err={train_tp_depth_err:.4f} "
                f"val_acc={val_acc:.4f} "
                f"val_iou={val_iou:.4f} "
                f"val_depth_err={val_depth_err:.4f} "
                f"val_tp_depth_err={val_tp_depth_err:.4f}"

                # f"train_acc={epoch_train_acc:.4f} "
                # f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
