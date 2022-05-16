import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

def dsc(y_pred, y_true, per_class=False):
    
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    
    if not per_class:
        return np.sum(y_pred[y_true == 1]) * 2 / (np.sum(y_pred) + np.sum(y_true))
    else:
        original_y_true = np.zeros_like(y_true).astype(int)
        original_y_pred = np.zeros_like(y_pred).astype(int)
        
        original_y_true[:, 0, ...] = y_true[:, 1, ...] - y_true[:, 2, ...]
        original_y_true[:, 1, ...] = y_true[:, 0, ...] - y_true[:, 1, ...]
        original_y_true[:, 2, ...] = y_true[:, 2, ...]
        
        original_y_pred[:, 0, ...] = y_pred[:, 1, ...] - y_pred[:, 2, ...]
        original_y_pred[:, 1, ...] = y_pred[:, 0, ...] - y_pred[:, 1, ...]
        original_y_pred[:, 2, ...] = y_pred[:, 2, ...]
        
        WT = np.sum(y_pred[:, 0, ...][y_true[:, 0, ...] == 1]) * 2 / (np.sum(y_pred[:, 0, ...]) + np.sum(y_true[:, 0, ...]) + 1e-8)
        TC = np.sum(y_pred[:, 1, ...][y_true[:, 1, ...] == 1]) * 2 / (np.sum(y_pred[:, 1, ...]) + np.sum(y_true[:, 1, ...]) + 1e-8)
        ET = np.sum(y_pred[:, 2, ...][y_true[:, 2, ...] == 1]) * 2 / (np.sum(y_pred[:, 2, ...]) + np.sum(y_true[:, 2, ...]) + 1e-8)
        
        NEC = np.sum(original_y_pred[:, 0, ...][original_y_true[:, 0, ...] == 1]) * 2 / (np.sum(original_y_pred[:, 0, ...]) + np.sum(original_y_true[:, 0, ...]) + 1e-8)
        ED = np.sum(original_y_pred[:, 1, ...][original_y_true[:, 1, ...] == 1]) * 2 / (np.sum(original_y_pred[:, 1, ...]) + np.sum(original_y_true[:, 1, ...]) + 1e-8)
    
        return WT, TC, ET, NEC, ED
    

def dsc_per_volume(validation_pred, validation_true, patient_slice_index, per_class=False):
    
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    if not per_class:
        dsc_list = []
        for p in range(len(num_slices)):        
            y_pred = np.array(validation_pred[index : index + num_slices[p]])
            y_true = np.array(validation_true[index : index + num_slices[p]])
            dsc_list.append(dsc(y_pred, y_true))
            index += num_slices[p]
        return dsc_list
    else:
        WT_list, TC_list, ET_list = [], [], []
        NET_list, ED_list = [], []
        for p in range(len(num_slices)):        
            y_pred = np.array(validation_pred[index : index + num_slices[p]])
            y_true = np.array(validation_true[index : index + num_slices[p]])
            WT, TC, ET, NET, ED = dsc(y_pred, y_true, per_class)
            WT_list.append(WT)
            TC_list.append(TC)
            ET_list.append(ET)
            NET_list.append(NET)
            ED_list.append(ED)
            index += num_slices[p]
        return  WT_list, TC_list, ET_list, NET_list, ED_list
    

def postprocess_per_volume(input_list, pred_list, true_list, patient_slice_index, patients):
    
    volumes = {}
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    
    for p in range(len(num_slices)):
        
        volume_in = np.array(input_list[index : index + num_slices[p]])
        volume_pred = np.round(np.array(pred_list[index : index + num_slices[p]])).astype(int)
        volume_true = np.array(true_list[index : index + num_slices[p]])
        
        volumes[patients[p]] = (volume_in, volume_pred, volume_true)
        index += num_slices[p]
        
    return volumes

def colour_labels(image, mask):
    tumour = 1 - np.sum(mask, axis=0)
    image[..., 0] = image[..., 1] = image[..., 2] = image[..., 0] * tumour
    image += 255 * mask.transpose(1, 2, 0).astype(np.uint8)
    return image


def gray2rgb(image):
    
    H, W = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((H, W, 3), dtype=np.uint8)
    ret[..., 2] = ret[..., 1] = ret[..., 0] = image * 255
    
    return ret
    

def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step + 1, prefix + "loss", np.mean(loss)))


def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step + 1, tag, value))


def log_class_summary(WT_value, TC_value, ET_value, step):
    print("epoch {} | mean_WT: {:4f}, mean_TC: {:4f}, mean_ET: {:4f}".format(step + 1, WT_value, TC_value, ET_value))


def log_class_orig_summary(NET_value, ED_value, step):
    print("epoch {} | mean_NEC: {:4f}, mean_ED: {:4f}".format(step + 1, NET_value, ED_value))

def show_metrics(label, metrics_array):
    print("\n{} Metrics:".format(label))
    print("Best Mean Precision: {:4f}".format(metrics_array[0]))
    print("Best Mean Sensitivity: {:4f}".format(metrics_array[1]))
    print("Best Mean Specificity: {:4f}".format(metrics_array[2]))
    print("Best Mean F1-Score: {:4f}".format(metrics_array[3]))
    print("Best Mean IoU: {:4f}".format(metrics_array[4]))
    
def Get_Metrics(y_pred, y_true):
    
    smooth = 1e-8 
    metrics = OrderedDict()
    
    # Get True Positives, True Negatives, False Positives, False Negatives.
    TP = np.sum(y_pred * y_true)
    TN = np.sum((1 - y_pred) * (1 - y_true))
    FP = np.sum((1 - y_true) * y_pred)
    FN = np.sum((1 - y_pred) * y_true)
    
    # Get Precision
    precision = (TP + smooth) / (TP + FP + smooth)
    metrics["Precision"] = precision
    
    # Get Sensitivity
    sensitivity = (TP + smooth) / (TP + FN + smooth)
    metrics["Sensitivity"] = sensitivity
    
    # Get Specificity
    specificity = (TN + smooth) / (TN + FP + smooth)
    metrics["Specificity"] = specificity
    
    # Get F1-Score / DSC
    F1_Score = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    metrics["F1_Score"] = F1_Score
    
    # Get IoU score
    IoU = (TP + smooth) / (TP + FP + FN + smooth)
    metrics["IoU"] = IoU
    
    return metrics


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)