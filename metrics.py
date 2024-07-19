import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
   
   Metrics-class to used for trained neural network, that outputs bounding box and binary class predictions. 
   -
   Metrics:
     * Precision: The accuracy of the positive predictions
     * Recall   : Ability of the model to detect all the relevant instances
     * MAP50    : Mean avg. precision at 50% IoU threshold
     * MAP50-95 : Mean avg. precision from 50% to 95% IoU threshold, with step of 5% (default)

   Usage:

   1st. Produce bunch of predictions, then proceed:
   
   Init:
   
   predictions   => Array of cornet bbox-corner coordinates e.g [[x1,y1,x2,y2], [x1,y1,x2,y2]...]
   ground_truths => Corresbonding ground truths, should match one with same index in predictions
   
   If no bbox coords. in image, mark with empty array: []



   No need to include anything in image specific, just bboxes.
   
   After init, call:
   
   metrics.precision or metrics.recall or mAP50 or metrics.mAP50_95
   
"""

# TODO: init vars to default via each func-call

class Metrics:
    def __init__(self, iou_threshold=0.5, step=0.05, ground_truths=[], predictions=[]):
        if len(ground_truths) != len(predictions):
            print("Ground truths and predictions differs in size")
            return 0
        self.iou_threshold = iou_threshold
        self.step = step
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.ious = []
        self.precision_val = 0.0
        self.recall_val = 0.0
        self.map50_val = 0.0
        self.map50_95_val = 0.0
        self.include_true_negative = False
        self.avoid_zero_division = 0.00001

    # Calculates intersection of single bounding-box-pair  
    def intersection(self,y_true, y_pred):
        (x1_t, y1_t, x2_t, y2_t) = y_true
        (x1_p, y1_p, x2_p, y2_p) = y_pred
        
        x1 = max(x1_t, x1_p)
        y1 = max(y1_t, y1_p)
        x2 = min(x2_t, x2_p)
        y2 = min(y2_t, y2_p)
        
        i_w = max(0, (x2-x1))
        i_h = max(0, (y2-y1))
        return i_w * i_h
        
    # Calculates union of single bounding-box-pair     
    def union(self, y_true, y_pred):
        (x1_t, y1_t, x2_t, y2_t) = y_true
        (x1_p, y1_p, x2_p, y2_p) = y_pred
        area_a = (x2_t - x1_t) * (y2_t - y1_t)
        area_b = (x2_p - x1_p) * (y2_p - y1_p)
        area_i = self.intersection(y_true, y_pred)
        return area_a + area_b - area_i
   
    # Calculates IoU of single bounding-box-pair (as its own function for transparency)
    def iou(self, intersection, union):
        return intersection / union
    
    def calculate_ious(self):
        for index, (y_true, y_pred) in enumerate(zip(self.ground_truths, self.predictions)):
            if y_true and y_pred:
                union = self.union(y_true, y_pred)
                intersection = self.intersection(y_true, y_pred)
                self.ious.append(self.iou(union, intersection))
            else:
                self.ious.append(0.0)
    
    # Sort (and append) single item to one of 4 possible c-matrix entries
    def sort_to_confusion_matrix(self, y_true, y_pred, iou):
    
        """
        This would be more reasonable in terms accuracy,
        but does not suit IoU thresholding:
        """
        if self.include_true_negative:
            if iou >= self.iou_threshold:
                self.tp += 1
            elif not y_true and y_pred: 
                self.fp += 1
            elif y_true and not y_pred:
                self.fn += 1
            elif not y_true and not y_pred:
               self.tn += 1
        else:
            if iou >= self.iou_threshold:
                self.tp += 1
            else:
                if y_pred:
                   self.fp += 1
                   self.fn += 1
   
    def calculate_confusion_matrix(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        
        self.calculate_ious()
        print('Len preds: ', len(self.predictions))
        for index, (y_true, y_pred, iou) in enumerate(zip(self.ground_truths, self.predictions, self.ious)):
            self.sort_to_confusion_matrix(y_true, y_pred, iou)
            
    def calculate_precision(self):
        precision = (self.tp * self.tp + self.fp) * 0.01
        self.precision_val = precision
    
    def calculate_recall(self):
        recall = (self.tp * self.tp + self.fn) * 0.01
        self.recall_val = recall
        
    def calculate_f1_score(self):
        f1_score = (2 * self.precision_val * self.recall_val) / (self.precision_val + self.recall_val)
        self.f1_score_val = f1_score
    
    def calculate_mAP50(self):
        self.calculate_ious()
        
        self.include_true_negative = False
        self.calculate_confusion_matrix()
        
        # Area under precision-recall curve
        auc = tf.keras.metrics.AUC(curve='PR')
        auc.update_state(self.recall_val, self.precision_val)
        ap = auc.result().numpy()
        self.map50_val = ap
    
    def calculate_mAP50_95(self):
        aps = []
        # Closest integer basically
        iters = int(0.45 / self.step)
        for i in range(iters):
            self.calculate_ious()
            self.calculate_precision()
            self.calculate_recall()
            # Area under precision-recall curve
            auc = tf.keras.metrics.AUC(curve='PR')
            auc.update_state(self.recall_val, self.precision_val)
            ap = auc.result()
            aps.append(ap.numpy())
            self.iou_threshold += self.step
        
        print("Aps", aps)
        self.map50_95_val = sum(aps) / iters
    
    # --------------------- "Getters" -----------------------------
    # Returns confusion matrix of: tp,fp,fn or tp,fp,fn,tn 
    def confusion_matrix(self, threshold, include_true_negative):
        if include_true_negative == 0:
            self.include_true_negative = False
        else:
            self.include_true_negative = True
        
        self.iou_threshold = threshold
        self.calculate_confusion_matrix()
        
        if self.include_true_negative:
            return self.tp, self.fp, self.fn, self.tn
        
        return self.tp, self.fp, self.fn
    
    # Returns precision, based on given IoU-threshold
    def precision(self, threshold):
        self.iou_threshold = threshold
        self.include_true_negative = False
        self.calculate_confusion_matrix()
        self.calculate_precision()
        return self.precision_val

    # Returns recall, based on given IoU-threshold
    def recall(self, threshold):
        self.iou_threshold = threshold
        self.include_true_negative = False
        self.calculate_confusion_matrix()
        self.calculate_recall()
        return self.recall_val
   
    # Returns f1-score, based on given IoU-threshold
    def f1_score(self, threshold):
        self.iou_threshold = threshold
        self.include_true_negative = False
        self.calculate_confusion_matrix()
        self.calculate_f1_score()
        return self.f1_score_val
   
    # Returns mean average precision with IoU threshold of 0.5     
    def mAP50(self):
        self.iou_threshold= 0.5
        self.calculate_mAP50()
        return self.map50_val

    # Returns mean average precision calculated from IoUs thresholded from 0.5 to 0.95 with step of 0.05       
    def mAP50_95(self, step):
        self.iou_threshold= 0.5
        self.step = step
        self.calculate_mAP50_95()
        return self.map50_95_val
