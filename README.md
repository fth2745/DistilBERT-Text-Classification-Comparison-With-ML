# DistilBERT-Text-Classification-Comparison-With-ML
Advanced text classification techniques using DistilBERT vectorization.

Dataset

The project reads data from a CSV file named MVSA. This file should be specified as an argument when running the code. The dataset should consist of texts and labels. Texts should be in the "text" column, and labels should be in the "label" column.

Model Selection and Feature Engineering

The project uses DistilBERT for text vectorization. This method converts texts into fixed-size vectors, allowing them to be used as input for machine learning models.

The following models are evaluated:

K-Nearest Neighbors (KNN)

Logistic Regression

Naive Bayes

Decision Tree

Support Vector Machines (SVM)

Artificial Neural Network (ANN)

Stacking Models:

(KNN, Logistic Regression, Naive Bayes)

(Decision Tree, SVM, ANN)

(KNN, Decision Tree, SVM)

Meta Stacking Model: (SVM, Stacking Model 3, Stacking Model 2)

Evaluation Metrics

The following metrics are used to evaluate the performance of the models:

Accuracy

Precision

Recall

F1-Score

AUC-ROC (if available)

Results Visualization

The performance results of the models are visualized with a bar graph. This graph shows the scores of different metrics for each model.

Additional Information

The code evaluates the performance of the models using 5-fold cross-validation.
Training and testing phases are performed separately for each model.
The results are stored in a Pandas DataFrame.
Matplotlib and Seaborn libraries are used for visualization.



![image](https://github.com/user-attachments/assets/a0d4818a-4fca-46a5-b508-4a0188dd4b2e)
