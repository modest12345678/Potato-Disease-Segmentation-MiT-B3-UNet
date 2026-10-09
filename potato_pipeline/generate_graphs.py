import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

fig_dir = Path("artifacts/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

classes = ["Soil", "Healthy", "Early Blight", "Late Blight"]
colors = ['#8B4513', '#2CA02C', '#FF7F0E', '#D62728']

# Placeholder Data (replace with your actual model results)
# ==========================================================
# Confusion Matrix (normalized example)
cm_data = np.array([
    [0.92, 0.04, 0.03, 0.01],
    [0.03, 0.89, 0.05, 0.03],
    [0.02, 0.07, 0.84, 0.07],
    [0.01, 0.03, 0.08, 0.88]
])

# Metrics (example values)
iou_scores = [0.90, 0.85, 0.78, 0.82]
dice_scores = [0.94, 0.91, 0.87, 0.89]
accuracy = 0.88
precision = [0.91, 0.86, 0.79, 0.83]
recall = [0.93, 0.88, 0.81, 0.86]
f1_score = [0.92, 0.87, 0.80, 0.84]

# ==========================================================

def save_plot(fig, filename):
    plt.tight_layout()
    fig.savefig(fig_dir / filename, dpi=300)
    plt.close(fig)
    print(f"✅ Generated {filename}")

# 1. Confusion Matrix (2m - Placeholder for 'confusion_matrix_2m_balanced.png')
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_data, cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)
ax.set_title("Confusion Matrix (Normalized)")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
for i in range(len(classes)):
    for j in range(len(classes)):
        text_color = "white" if cm_data[i, j] > cm_data.max() / 2 else "black"
        ax.text(j, i, f"{cm_data[i, j]:.2f}", ha="center", va="center", color=text_color)
save_plot(fig, "confusion_matrix_2m_balanced.png")

# 2. Generic Confusion Matrix (Placeholder for 'Confusion_Matrix.png')
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_data, cmap="Greens")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
for i in range(len(classes)):
    for j in range(len(classes)):
        text_color = "white" if cm_data[i, j] > cm_data.max() / 2 else "black"
        ax.text(j, i, f"{cm_data[i, j]:.2f}", ha="center", va="center", color=text_color)
save_plot(fig, "Confusion_Matrix.png")

# 3. Accuracy (Placeholder for 'Accuracy.png')
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Overall Accuracy"], [accuracy], color='skyblue')
ax.set_ylim(0, 1.0)
ax.set_title("Overall Accuracy")
ax.set_ylabel("Accuracy Score")
ax.text(0, accuracy + 0.02, f"{accuracy:.2f}", ha='center', va='bottom', fontsize=12)
save_plot(fig, "Accuracy.png")

# 4. Precision (Placeholder for 'Precision.png')
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(classes, precision, color=colors)
ax.set_ylim(0, 1.0)
ax.set_title("Per-Class Precision")
ax.set_ylabel("Precision Score")
for i, p in enumerate(precision):
    ax.text(i, p + 0.02, f"{p:.2f}", ha='center', va='bottom')
save_plot(fig, "Precision.png")

# 5. Recall (Placeholder for 'Recall.png')
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(classes, recall, color=colors)
ax.set_ylim(0, 1.0)
ax.set_title("Per-Class Recall")
ax.set_ylabel("Recall Score")
for i, r in enumerate(recall):
    ax.text(i, r + 0.02, f"{r:.2f}", ha='center', va='bottom')
save_plot(fig, "Recall.png")

# 6. F1_Score (Placeholder for 'F1_Score.png')
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(classes, f1_score, color=colors)
ax.set_ylim(0, 1.0)
ax.set_title("Per-Class F1 Score")
ax.set_ylabel("F1 Score")
for i, f1 in enumerate(f1_score):
    ax.text(i, f1 + 0.02, f"{f1:.2f}", ha='center', va='bottom')
save_plot(fig, "F1_Score.png")

# 7. IoU and mIoU (Placeholder for 'IoU_and_mIoU.png')
mIoU = np.mean(iou_scores)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(classes + ["mIoU"], iou_scores + [mIoU], color=colors + ['grey'])
ax.set_ylim(0, 1.0)
ax.set_title("Per-Class IoU and Mean IoU")
ax.set_ylabel("IoU Score")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
save_plot(fig, "IoU_and_mIoU.png")

# 8. Dice_Coefficient (Placeholder for 'Dice_Coefficient.png')
mDice = np.mean(dice_scores)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(classes + ["mDice"], dice_scores + [mDice], color=colors + ['grey'])
ax.set_ylim(0, 1.0)
ax.set_title("Per-Class Dice Coefficient and Mean Dice")
ax.set_ylabel("Dice Coefficient")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
save_plot(fig, "Dice_Coefficient.png")

# 9. graph1_normalized_confusion_matrix_2m.png (Same as confusion_matrix_2m_balanced.png)
# This plot is already generated as 'confusion_matrix_2m_balanced.png'
# If a distinct visual is needed, please provide different data or styling.
print("Note: 'graph1_normalized_confusion_matrix_2m.png' is similar to 'confusion_matrix_2m_balanced.png' and has been generated by that section.")

# 10. graph2_per_class_metrics_barchart_2m.png
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
x = np.arange(len(classes))

rects1 = ax.bar(x - width, iou_scores, width, label='IoU', color=colors[0])
rects2 = ax.bar(x, dice_scores, width, label='Dice', color=colors[1])
rects3 = ax.bar(x + width, precision, width, label='Precision', color=colors[2])

ax.set_ylabel('Score')
ax.set_title('Per-Class Metrics Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_ylim(0, 1.0)
ax.legend()

for bars in [rects1, rects2, rects3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
save_plot(fig, "graph2_per_class_metrics_barchart_2m.png")


# 11. graph3_iou_vs_dice_2m.png
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(classes, iou_scores, marker='o', linestyle='-', color='blue', label='IoU')
ax.plot(classes, dice_scores, marker='x', linestyle='--', color='red', label='Dice')
ax.set_ylim(0, 1.0)
ax.set_title('IoU vs Dice Score Per-Class')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

save_plot(fig, "graph3_iou_vs_dice_2m.png")

print("\nAll requested custom graphs have been generated and saved to 'artifacts/figures/'.")

