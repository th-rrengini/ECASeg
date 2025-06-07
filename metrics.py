import torch
import gc
import numpy as np
from torch.nn.functional import sigmoid
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import json
from torchmetrics.segmentation import MeanIoU
from cityscapes_ids import labels


gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(model, imgs, masks, loss_fn):
    imgs = imgs.to(device)
    masks = masks.to(device)
    output = model(imgs)
    del imgs
    loss = loss_fn(output, masks)
    return loss, output


def compute_accuracy(predicted, targets):
    predicted = predicted.to('cpu')
    targets = targets.to('cpu')

    # Count correct predictions
    correct_predictions = torch.eq(predicted, targets).sum().item()
   
    # Total number of predictions
    total_predictions = targets.numel()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy

def compute_recall(pred, target):
    pred = pred.to(device)
    target = target.to(device)
    # True Positives: predicted 1, actually 1
    TP = ((pred.round() == 1) & (target == 1)).sum().float()
    
    # False Negatives: predicted 0, actually 1
    FN = ((pred.round() == 0) & (target == 1)).sum().float()
    
    # Avoid division by zero
    if TP + FN == 0:
        recall = torch.tensor(0.0)
    else:
        recall = TP / (TP + FN)
    return recall

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def compute_instance_iou(hist, label_trues, label_preds, n_class, ignore_class):
    # Initialize dictionaries to hold instance-wise counts
    instance_sizes = {i: [] for i in range(n_class)}  # Collect sizes of instances for each class
    weighted_tp = {i: 0 for i in range(n_class)}
    weighted_fn = {i: 0 for i in range(n_class)}
    fp = {i: 0 for i in range(n_class)}

    for lt, lp in zip(label_trues, label_preds):
        for class_id in range(n_class):
            if class_id == ignore_class:
                continue
            
            # Collect instance sizes for later calculation (size of ground truth instances)
            instance_size = np.sum(lt == class_id)
            if instance_size > 0:
                instance_sizes[class_id].append(instance_size)
                
            # True positives, false positives, false negatives
            tp_pixels = np.sum((lp == class_id) & (lt == class_id))
            fn_pixels = np.sum((lp != class_id) & (lt == class_id))
            fp_pixels = np.sum((lp == class_id) & (lt != class_id))
            
            if instance_size > 0:
                # Calculate weighted TP, FN based on instance size
                avg_instance_size = np.mean(instance_sizes[class_id]) if instance_sizes[class_id] else 1
                weight = avg_instance_size / instance_size
                weighted_tp[class_id] += tp_pixels * weight
                weighted_fn[class_id] += fn_pixels * weight
            
            fp[class_id] += fp_pixels

    # Calculate iIoU for each class
    iIoU = {
        class_id: weighted_tp[class_id] / (weighted_tp[class_id] + fp[class_id] + weighted_fn[class_id])
        for class_id in range(n_class)
        if class_id != ignore_class and (weighted_tp[class_id] + fp[class_id] + weighted_fn[class_id]) > 0
    }

    # Compute the mean instance IoU for all valid classes
    valid_iIoU = [iIoU[c] for c in range(n_class) if c in iIoU]
    mean_instance_iou = np.nanmean(valid_iIoU) if valid_iIoU else float('nan')

    return mean_instance_iou



def scores(label_trues, label_preds, n_class, ignore_class):
    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    # Compute instance IoU (add your implementation of this function if needed)
    mean_instance_iou = compute_instance_iou(hist, label_trues, label_preds, n_class, ignore_class)
    
    # Pixel Accuracy
    acc = np.diag(hist).sum() / hist.sum()
    
    # Mean Accuracy
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    
    # IoU calculation
    divisor = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iu = np.diag(hist) / divisor
    
    # Handle multiple ignored classes
    valid = (hist.sum(axis=1) > 0) & (~np.isin(np.arange(n_class), ignore_class))  # Ignore multiple classes
    mean_iu = np.nanmean(iu[valid])
    
    # Frequency Weighted IoU
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Mean Instance IoU": mean_instance_iou,
        "Class IoU": cls_iu,
    }


def compute_Binary_miou(pred, target, miou_metric):   
    pred = pred.to('cpu') 
    target = target.to('cpu')
    pred = pred.squeeze(1)  
    target = target.squeeze(1)
    return miou_metric(pred, target)

def compute_BCEloss(model, imgs, masks, loss_fn):
    imgs = imgs.to(device)
    masks = masks.to(device)

    output = model(imgs)
    del imgs
    loss = loss_fn(output, masks)
    return loss, output

