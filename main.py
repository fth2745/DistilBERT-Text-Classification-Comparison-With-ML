from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data = pd.read_csv('your_data')

# Text and Label Columns
text_column = 'text'
label_column = 'label'

# Prepare Data
corpus = data[text_column].fillna("").values
labels = data[label_column].values

# DistilBERT Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Text to BERT Embedding Function
def text_to_bert_embedding(texts):
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Convert Texts to Vectors
X = text_to_bert_embedding(corpus)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Models
knn = KNeighborsClassifier()
lr = LogisticRegression(max_iter=1000, random_state=42)
nb = GaussianNB()
dt = DecisionTreeClassifier()
svm = SVC(probability=True)
ysa = MLPClassifier(max_iter=1000, random_state=42)

# Stacking Models
stacking_model_1 = StackingClassifier(estimators=[('knn', knn), ('lr', lr), ('nb', nb)], final_estimator=LogisticRegression())
stacking_model_2 = StackingClassifier(estimators=[('dt', dt), ('svm', svm), ('ysa', ysa)], final_estimator=RandomForestClassifier(random_state=42))
stacking_model_3 = StackingClassifier(estimators=[('knn', knn), ('dt', dt), ('svm', svm)], final_estimator=GradientBoostingClassifier(random_state=42))
meta_stacking_model = StackingClassifier(estimators=[('svm', svm), ('stacking3', stacking_model_3), ('stacking2', stacking_model_2)], final_estimator=RandomForestClassifier(random_state=42))

# Model Dictionary
models = {'KNN': knn, 'Logistic Regression': lr, 'Naive Bayes': nb, 'Decision Tree': dt, 'SVM': svm, 'ANN': ysa, 
          'Stacking Model 1': stacking_model_1, 'Stacking Model 2': stacking_model_2, 'Stacking Model 3': stacking_model_3, 
          'Meta Stacking Model': meta_stacking_model}

# Performance Results
results = []

# Train and Evaluate Models
for name, model in tqdm(models.items(), desc="Training and Evaluating Models"):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        auc_roc = None

    results.append({'Model': name, 'CV Mean Accuracy': cv_scores.mean(), 'CV Std Dev': cv_scores.std(),
                    'Test Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC-ROC': auc_roc})

# Convert Results to DataFrame
results_df = pd.DataFrame(results)

# Prepare DataFrame for Visualization
results_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# Plot
plt.figure(figsize=(14, 8))
ax = sns.barplot(data=results_df, x='Model', y='Score', hue='Metric', palette='viridis')

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9, padding=3)

plt.title("Model Performance Comparison", fontsize=16)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Metrics", fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.show()