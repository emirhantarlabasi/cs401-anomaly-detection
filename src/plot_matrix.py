import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array([[9071, 929], 
               [4068, 5932]])

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)

#Heatmap 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'],
            linewidths=1, linecolor='black')

plt.title('Confusion Matrix - Isolation Forest (Test Set)', fontsize=16)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)


plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=300)
print("✅ Grafik oluşturuldu: confusion_matrix_final.png")
plt.show()