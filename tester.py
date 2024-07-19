from metrics import *

"""
100%
Should produce(mAP50): 1.0
Matched?: true
"""

"""
# List of ground truth bounding-boxes (x1, y1, x2, y2)
ground_truths = [
    [0.64, 0.23, 0.87, 0.67],
    [0.15, 0.10, 0.45, 0.40],
    [0.50, 0.50, 0.75, 0.80],
    [0.30, 0.30, 0.60, 0.55],
    [0.05, 0.05, 0.25, 0.25],
    [0.70, 0.70, 0.90, 0.90],
    [0.40, 0.20, 0.60, 0.45],
    [0.25, 0.60, 0.50, 0.85],
    [0.10, 0.50, 0.35, 0.75],
    [0.55, 0.10, 0.80, 0.35],
]

# Corresponding pairs, identical in this case
predictions = [
    [0.64, 0.23, 0.87, 0.67],
    [0.15, 0.10, 0.45, 0.40],
    [0.50, 0.50, 0.75, 0.80],
    [0.30, 0.30, 0.60, 0.55],
    [0.05, 0.05, 0.25, 0.25],
    [0.70, 0.70, 0.90, 0.90],
    [0.40, 0.20, 0.60, 0.45],
    [0.25, 0.60, 0.50, 0.85],
    [0.10, 0.50, 0.35, 0.75],
    [0.55, 0.10, 0.80, 0.35],
]

metr = Metrics(ground_truths=ground_truths, predictions=predictions)

(true_positives, false_positives, false_negatives) = metr.confusion_matrix(0.5, 0)
print('Confusion matrix(IoU th only): ', true_positives, '  ', false_positives, '  ', false_negatives)

(true_positives, false_positives, false_negatives, true_negatives) = metr.confusion_matrix(0.5, 1)
print('Confusion matrix(with tns):', true_positives, '  ', false_positives, '  ', false_negatives, '  ',  true_negatives)

precision = metr.precision(0.5)
print('Precision: ', precision)

recall = metr.recall(0.5)
print('Recall: ', recall)

f1_score = metr.f1_score(0.5)
print('F1-score: ', f1_score)

mAP50 = metr.mAP50()
print('mAP50: ', mAP50)

mAP50_95 = metr.mAP50_95(0.05)
print('mAP50-95: ', mAP50_95)
"""

"""
75%
Should produce(mAP50): 0.75
Matched?:
"""

# Error of 25% to predictions bottom-right coordinates
for i in range(len(predictions)):

metr = Metrics(ground_truths=ground_truths, predictions=predictions)

(true_positives, false_positives, false_negatives) = metr.confusion_matrix(0.5, 0)
print('Confusion matrix(IoU th only): ', true_positives, '  ', false_positives, '  ', false_negatives)

(true_positives, false_positives, false_negatives, true_negatives) = metr.confusion_matrix(0.5, 1)
print('Confusion matrix(with tns):', true_positives, '  ', false_positives, '  ', false_negatives, '  ',  true_negatives)

precision = metr.precision(0.5)
print('Precision: ', precision)

recall = metr.recall(0.5)
print('Recall: ', recall)

f1_score = metr.f1_score(0.5)
print('F1-score: ', f1_score)

mAP50 = metr.mAP50()
print('mAP50: ', mAP50)

mAP50_95 = metr.mAP50_95(0.05)
print('mAP50-95: ', mAP50_95)


"""
50%
Should produce(mAP50): 0.5
Matched?:
"""

"""

metr = Metrics(ground_truths=ground_truths, predictions=predictions)

(true_positives, false_positives, false_negatives) = metr.confusion_matrix(0.5, 0)
print('Confusion matrix(IoU th only): ', true_positives, '  ', false_positives, '  ', false_negatives)

(true_positives, false_positives, false_negatives, true_negatives) = metr.confusion_matrix(0.5, 1)
print('Confusion matrix(with tns):', true_positives, '  ', false_positives, '  ', false_negatives, '  ',  true_negatives)

precision = metr.precision(0.5)
print('Precision: ', precision)

recall = metr.recall(0.5)
print('Recall: ', recall)

f1_score = metr.f1_score(0.5)
print('F1-score: ', f1_score)

mAP50 = metr.mAP50()
print('mAP50: ', mAP50)

mAP50_95 = metr.mAP50_95(0.05)
print('mAP50-95: ', mAP50_95)

"""

"""
23%
Should produce(mAP50): 0.0
Matched?:
"""

"""

metr = Metrics(ground_truths=ground_truths, predictions=predictions)

(true_positives, false_positives, false_negatives) = metr.confusion_matrix(0.5, 0)
print('Confusion matrix(IoU th only): ', true_positives, '  ', false_positives, '  ', false_negatives)

(true_positives, false_positives, false_negatives, true_negatives) = metr.confusion_matrix(0.5, 1)
print('Confusion matrix(with tns):', true_positives, '  ', false_positives, '  ', false_negatives, '  ',  true_negatives)

precision = metr.precision(0.5)
print('Precision: ', precision)

recall = metr.recall(0.5)
print('Recall: ', recall)

f1_score = metr.f1_score(0.5)
print('F1-score: ', f1_score)

mAP50 = metr.mAP50()
print('mAP50: ', mAP50)

mAP50_95 = metr.mAP50_95(0.05)
print('mAP50-95: ', mAP50_95)

"""

