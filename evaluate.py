"""Train on data/train.csv, evaluate on the 15 unseen titles in data/test.csv."""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from classifier import train

model = train()
test = pd.read_csv("data/test.csv")
pred = model.predict(test["title"])

print(f"Accuracy: {accuracy_score(test['category'], pred):.1%} "
      f"({(pred == test['category']).sum()}/{len(test)})\n")
print(classification_report(test["category"], pred, zero_division=0))
print("Misclassified:")
for t, y, p in zip(test["title"], test["category"], pred):
    if y != p:
        print(f"  - {t}\n    true={y} | predicted={p}")
