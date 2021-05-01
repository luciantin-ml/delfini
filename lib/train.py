from dolphins_recognition_challenge.instance_segmentation.model import show_predictions, iou_metric
from dolphins_recognition_challenge import utils
import optuna
import math
import sys


def train_model(trial, num_epochs, model, optimizer, scheduler, device, data_loader, data_loader_test, ):
    # metrics = []
    mean_iou_testset = 0

    for epoch in range(num_epochs):
        print(f"Epoch #{epoch}")

        # train for 1 epoch, has model.train()
        metric = train_one_epoch(model, optimizer, data_loader, device, epoch=epoch, print_freq=1)
        print("Metrics", metric)
        # metrics.append(metric)

        # has model.eval()
        mean_iou_testset, _ = iou_metric(model, data_loader_test.dataset)
        print(f"Mean IOU metric for the test set is: {mean_iou_testset}")

        # -||-
        # show_predictions(model, data_loader=data_loader_test, n=5, score_threshold=0.5)
        trial.report(mean_iou_testset, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # update learning rate
        scheduler.step()

    return mean_iou_testset


def train_one_epoch(
        model,
        optimizer,
        data_loader,
        device,
        epoch,
        print_freq=10,
):
    """ Trains one epoch of the model. Copied from the reference implementation from https://github.com/pytorch/vision.git. """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger
