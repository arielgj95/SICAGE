import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

save_path = Path("analysis_outputs/classifier_analysis")
save_path.mkdir(parents=True, exist_ok=True)
# Test F1 scores for adversarial learning
adversarial_f1_scores = {
    "mot": 0.3939828546369154,
    "w2v": 0.9814048164802299,
    "text": 0.8681393480988904,
    "ons + mels": 0.6412289696552554,
    "ons + mels + w2v": 0.9805589077491187,
    "text + ons + mels + w2v": 0.9740776213407318,
    "mot + text + ons + mels + w2v": 0.9836938928595548,
}

# Test F1 scores for standard learning
standard_f1_scores = {
    "mot": 0.3852657478836593,
    "w2v": 0.9839860602128737,
    "text": 0.8421782938714261,
    "ons + mels": 0.5705277135518972,
    "ons + mels + w2v": 0.9805589077491187,
    "text + ons + mels + w2v": 0.9698531443354107,
    "mot + text + ons + mels + w2v": 0.9799316627418138,
}

# Classifier labels
classifiers = list(adversarial_f1_scores.keys())

# Data for plotting
adversarial = list(adversarial_f1_scores.values())
standard = list(standard_f1_scores.values())

# Plotting the histogram
x = np.arange(len(classifiers))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, adversarial, width, label="Adversarial Learning")
plt.bar(x + width / 2, standard, width, label="Standard Learning")

# Adding labels, title, and legend
plt.xticks(x, classifiers, rotation=45, ha="right")
plt.xlabel("Classifiers")
plt.ylabel("Test F1 Score")
plt.title("Test F1 Scores Across Classifiers")
plt.legend()

# Showing the plot
plt.tight_layout()
save_fig = save_path / "classifiers_results_ted4c.png"
plt.savefig(save_fig )
plt.show()



# Data for F1 scores
classifiers = [
    "Subj-dep test Standard",
    "Subj-indep test Standard",
    "Subj-indep test Adversarial"
]
f1_scores = [
    0.8238298947365695,  # Subj-dep test Standard
    0.3852657478836593,  # Subj-indep test Standard
    0.3939828546369154   # Subj-indep test Adversarial
]

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, f1_scores, color=['skyblue', 'salmon', 'lightgreen'], edgecolor='black')

# Add labels to the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha='center', va='bottom', fontsize=12)

# Labels and title
plt.title('Test F1 Scores of Motion Classifiers', fontsize=16)
plt.ylabel('F1 Score', fontsize=14)
plt.xlabel('Classifiers', fontsize=14)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig(save_path / "culclA_ted4c.png")
plt.show()
