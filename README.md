# **Credit Approval Analysis and Prediction**

## **Overview**
This project explores and predicts credit approval outcomes using a combination of unsupervised and supervised machine learning techniques. The analysis identifies patterns, segments applicants, and builds predictive models to inform credit risk assessment.

---

## **Key Features**
### **Unsupervised Analysis**
- **Principal Component Analysis (PCA)**:
  - Reduced dimensionality and identified key factors driving variance.
- **Hierarchical Clustering**:
  - Segmented the dataset into 4 clusters based on financial and demographic characteristics.
- **t-SNE and UMAP**:
  - Visualized high-dimensional data to reveal clusters and global structure.

### **Supervised Analysis**
- Compared machine learning models for credit approval prediction:
  - **Logistic Regression (Elastic Net)**: Feature selection and interpretability.
  - **Random Forest**: Variable importance and non-linear relationships.
  - **XGBoost**: High performance with hyperparameter tuning.
  - **Neural Network**: Competitive accuracy with a simple architecture.

---

## **Results**
### **Unsupervised Analysis**
- Identified 4 distinct clusters representing subpopulations with unique financial and demographic profiles.
- Visualizations revealed overlaps and separations in applicant characteristics.

### **Supervised Analysis**
- Best accuracy achieved by:
  - **Neural Network**: 88.41%
  - **XGBoost**: 86.96%
- Logistic regression provided the highest interpretability, selecting **Prior Default**, **Credit Score**, and **Income** as key predictors.
