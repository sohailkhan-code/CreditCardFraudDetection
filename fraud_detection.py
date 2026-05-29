# ============================================
# CREDIT CARD FRAUD DETECTION
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   CREDIT CARD FRAUD DETECTION PROJECT")
print("=" * 60)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n[1/6] Loading dataset...")
df = pd.read_csv('creditcard.csv')

print(f"✅ Dataset loaded!")
print(f"   Total transactions : {len(df)}")
print(f"   Fraudulent cases   : {df['is_fraud'].sum()}")
print(f"   Legitimate cases   : {(df['is_fraud'] == 0).sum()}")

# ============================================
# STEP 2: EXPLORE & CLEAN DATA
# ============================================
print("\n[2/6] Exploring data...")

# Class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x='is_fraud', data=df, palette=['green', 'red'])
plt.title('Transaction Distribution\n(0 = Legitimate, 1 = Fraud)')
plt.xlabel('is_fraud')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()
print("✅ Graph saved: class_distribution.png")

# ============================================
# STEP 3: PREPARE DATA
# ============================================
print("\n[3/6] Preparing data...")

# Drop columns jo useful nahi hain
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num',
             'first', 'last', 'street', 'trans_num', 'dob']
df = df.drop(columns=drop_cols)

# Categorical columns ko numbers mein convert karo
cat_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features aur target alag karo
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Scale karo
scaler = StandardScaler()
X[['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time',
   'merch_lat', 'merch_long']] = scaler.fit_transform(
    X[['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time',
       'merch_lat', 'merch_long']])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE — class balance karo
print("   Applying SMOTE (thoda time lagega)...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE - Training samples: {len(X_train_sm)}")

# ============================================
# STEP 4: TRAIN 3 MODELS
# ============================================
print("\n[4/6] Training models...")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100,
                                                    random_state=42,
                                                    n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}... (wait karo)")
    model.fit(X_train_sm, y_train_sm)
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    results[name] = {'model': model, 'y_pred': y_pred,
                     'accuracy': acc, 'roc_auc': roc_auc}
    print(f"   ✅ {name} | Accuracy: {acc*100:.2f}% | ROC-AUC: {roc_auc:.4f}")

# ============================================
# STEP 5: RESULTS & GRAPHS
# ============================================
print("\n[5/6] Generating results...")

for name, res in results.items():
    print(f"\n📊 {name}")
    print("-" * 40)
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['Legitimate', 'Fraud']))

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_title(f'{name}\nAcc: {res["accuracy"]:.3f}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()
print("✅ Graph saved: confusion_matrices.png")

# Model comparison
model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
roc_scores  = [results[m]['roc_auc']  for m in model_names]

x     = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, roc_scores,  width, label='ROC-AUC',  color='coral')
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("✅ Graph saved: model_comparison.png")

# ============================================
# STEP 6: BEST MODEL
# ============================================
print("\n[6/6] Best Model...")
best = max(results, key=lambda x: results[x]['roc_auc'])
print(f"\n🏆 BEST MODEL: {best}")
print(f"   ROC-AUC: {results[best]['roc_auc']:.4f}")

print("\n" + "=" * 60)
print("✅ PROJECT COMPLETE!")
print("   📊 class_distribution.png")
print("   📊 confusion_matrices.png")
print("   📊 model_comparison.png")
print("=" * 60)