# 📊 Telco Customer Churn Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit)](https://customer-churn-prediction.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge)]()

> **Machine Learning solution untuk memprediksi customer churn di industri telekomunikasi dengan akurasi tinggi, membantu perusahaan menghemat hingga 23,71% biaya promosi.**

## 🎯 Demo Aplikasi

🔗 **[Customer Churn Prediction - Streamlit App](https://customer-churn-prediction.streamlit.app)**

Aplikasi prediksi churn

---

## 📋 Daftar Isi

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Business Impact](#-business-impact)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Results & Insights](#-results--insights)
- [Recommendations](#-recommendations)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)

---

## 🎯 Overview

Proyek ini mengembangkan sistem prediksi customer churn berbasis Machine Learning untuk perusahaan telekomunikasi. Model ini mampu mengidentifikasi pelanggan yang berisiko tinggi untuk berhenti berlangganan, sehingga perusahaan dapat melakukan intervensi tepat waktu dengan strategi retensi yang lebih efektif dan efisien.

### Key Features

✅ Prediksi churn dengan **F2-Score 0.745** (setelah tuning)  
✅ Pengurangan kerugian finansial hingga **23,71%** ($7,130 → $5440)  
✅ Interactive dashboard dengan **Streamlit**  
✅ Explainable AI menggunakan **LIME & SHAP**  
✅ Feature importance analysis untuk business insights  

---

## 💼 Business Problem

### Latar Belakang

Perusahaan telekomunikasi menghadapi tantangan besar dalam mempertahankan pelanggan setia di tengah persaingan industri yang kompetitif. Tingginya tingkat customer churn berpotensi menimbulkan kerugian finansial signifikan karena:

- **Strategi promosi tidak efisien**: Promosi dilakukan secara massal tanpa mempertimbangkan pelanggan yang benar-benar berisiko
- **Biaya promosi membengkak**: Pemborosan hingga $71,300 untuk promosi ke pelanggan yang sebenarnya loyal
- **Kehilangan pelanggan bernilai tinggi**: Tidak ada sistem early warning untuk deteksi risiko churn

### Problem Statement

Perusahaan membutuhkan sistem prediksi churn berbasis Machine Learning untuk:

1. **Mendeteksi lebih awal** pelanggan yang berpotensi churn
2. **Mengoptimalkan strategi retensi** melalui promosi yang tepat sasaran
3. **Mengurangi pemborosan biaya promosi** dan meningkatkan profitabilitas

### Cost Analysis

| Jenis Kesalahan | Biaya | Dampak |
|----------------|-------|--------|
| **False Positive (FP)** | $10 | Promosi diberikan ke pelanggan loyal (pemborosan) |
| **False Negative (FN)** | $80 | Kehilangan pelanggan berisiko tinggi |

**Prioritas**: Meminimalkan False Negative (FN) karena cost-nya 8x lebih besar

---

## 📊 Dataset

### Dataset Information

- **Total Records**: 4,853 (setelah menghapus duplikat dari 4,930)
- **Features**: 10 features + 1 target variable
- **Target Distribution**: 
  - No Churn: 73.3% (3,614 pelanggan)
  - Churn: 26.7% (1,316 pelanggan)
- **Class Imbalance**: Ya (ditangani dengan SMOTE & RandomOverSampler)

### Features Description

#### Categorical Features (8)
- **Dependents**: Status tanggungan keluarga
- **OnlineSecurity**: Status layanan keamanan online
- **OnlineBackup**: Status layanan backup online
- **InternetService**: Jenis layanan internet (DSL/Fiber optic/No)
- **DeviceProtection**: Status perlindungan perangkat
- **TechSupport**: Status layanan dukungan teknis
- **Contract**: Jenis kontrak (Month-to-month/One year/Two year)
- **PaperlessBilling**: Penggunaan tagihan digital

#### Numerical Features (2)
- **tenure**: Lama berlangganan (dalam bulan)
- **MonthlyCharges**: Biaya bulanan pelanggan

#### Target Variable
- **Churn**: Status churn pelanggan (0 = No, 1 = Yes)

---

## 🔬 Methodology

### 1. Data Preprocessing

```python
# Handling
✅ Duplikasi data (77 rows removed)
✅ Missing values (tidak ditemukan)
✅ Feature encoding (One-Hot Encoding)
✅ Feature scaling (RobustScaler)
```

### 2. Exploratory Data Analysis (EDA)

**Key Findings:**
- Pelanggan dengan **tenure pendek** (<6 bulan) memiliki risiko churn tinggi
- **Kontrak bulanan** lebih rentan churn dibanding kontrak tahunan
- **Fiber optic users** lebih cenderung churn
- **MonthlyCharges tinggi** berkorelasi dengan churn

### 3. Feature Engineering

- **One-Hot Encoding** untuk categorical features
- **RobustScaler** untuk numerical features (tahan terhadap outlier)
- **SMOTE & RandomOverSampler** untuk handling imbalanced data

### 4. Model Selection & Training

**Models Tested:**
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- **Gradient Boosting Classifier** ✅ (Best Model)
- XGBoost

**Evaluation Metric**: **F2-Score** (memprioritaskan recall)

### 5. Hyperparameter Tuning

**RandomizedSearchCV** dengan 1000 iterasi:
- `max_depth`: 1
- `learning_rate`: 0.34
- `n_estimators`: 93
- `subsample`: 0.6
- `max_features`: 3
- `resampler`: RandomOverSampler

---

## 📈 Model Performance

### Performance Metrics

| Metric | Before Tuning | After Tuning | Improvement |
|--------|---------------|--------------|-------------|
| **F2-Score (Train)** | 0.702 | 0.730 | +0.028 |
| **F2-Score (Test)** | 0.719 | 0.745 | +0.026 |
| **Recall (Churn)** | 0.78 | 0.83 | +0.05 |
| **Precision (Churn)** | 0.55 | 0.54 | -0.01 |
| **Accuracy** | 0.77 | 0.76 | -0.01 |

### Confusion Matrix (After Tuning)

|  | Predicted No Churn | Predicted Churn |
|--|-------------------|-----------------|
| **Actual No Churn** | 529 | 184 |
| **Actual Churn** | 45 | 213 |

**Key Improvements:**
- ✅ False Negative turun dari 57 → 45 (21% reduction)
- ✅ True Positive naik dari 201 → 213 (6% increase)
- ⚠️ False Positive naik dari 164 → 184 (trade-off yang acceptable)

### Feature Importance

**Top 5 Most Important Features:**

1. **InternetService_Fiber optic** (0.256) - Tipe layanan internet
2. **Contract_Two year** (0.236) - Kontrak 2 tahun
3. **tenure** (0.227) - Lama berlangganan
4. **Contract_One year** (0.089) - Kontrak 1 tahun
5. **DeviceProtection_No internet service** (0.044)

---

## 💰 Business Impact

### Cost Comparison

#### ❌ Without Machine Learning
- **Total Promotion Cost**: 971 customers × $10 = **$9,710**
- **Wasted Promotion** (ke pelanggan loyal): 713 × $10 = **$7,130**
- **Effective Promotion**: 258 × $10 = $2,580

#### ✅ With Machine Learning
- **False Positive Cost**: 184 × $10 = $1,840
- **False Negative Cost**: 45 × $80 = $3,600
- **Total Loss**: **$5,440**

### 💡 Savings Achieved

```
Kerugian Berkurang: $7,130 → $5,440
Penghematan: $1,660
Persentase Penghematan: 23,71%
```

**ROI**: Machine Learning berhasil mengurangi kerugian hingga **23,71**! 🎉

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/username/telco-churn-prediction.git
cd telco-churn-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```txt
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
xgboost==1.7.6
category-encoders==2.6.1
streamlit==1.28.0
lime==0.2.0.1
shap==0.43.0
dill==0.3.7
```

---

## 💻 Usage

### Running the Streamlit App

```bash
# Local development
streamlit run app.py

# The app will open in your browser at http://localhost:8501
```

### Using the Model in Python

```python
import pickle
import pandas as pd

# Load model
model = pickle.load(open('model_gbc.sav', 'rb'))

# Prepare input data
data = {
    'Dependents': 'Yes',
    'tenure': 9,
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'InternetService': 'DSL',
    'DeviceProtection': 'Yes',
    'TechSupport': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'MonthlyCharges': 72.90
}

df = pd.DataFrame([data])

# Make prediction
prediction = model.predict(df)
probability = model.predict_proba(df)

print(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Churn Probability: {probability[0][1]:.2%}")
```

---

## 🛠 Technologies

### Machine Learning
- **scikit-learn**: Model training & evaluation
- **XGBoost**: Gradient boosting implementation
- **imbalanced-learn**: Handling class imbalance
- **category_encoders**: Feature encoding

### Data Analysis & Visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib & Seaborn**: Data visualization

### Explainable AI
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations

### Deployment
- **Streamlit**: Interactive web application
- **Pickle**: Model serialization

---

## 🔍 Results & Insights

### Key Insights from Analysis

#### 1. **Tenure adalah Faktor Terpenting**
- Pelanggan baru (< 6 bulan) memiliki churn rate tertinggi
- Lonjakan churn signifikan terjadi di **bulan pertama**
- Pelanggan dengan tenure > 60 bulan sangat loyal

#### 2. **Contract Type Matters**
- **Month-to-month**: Churn rate tinggi
- **One year**: Churn rate sedang
- **Two year**: Churn rate rendah (pelanggan paling loyal)

#### 3. **Internet Service Impact**
- **Fiber optic users** lebih banyak churn
- Kemungkinan: ekspektasi tinggi vs kualitas layanan
- Perlu improvement pada layanan fiber

#### 4. **Support Services**
- Pelanggan dengan **TechSupport** lebih loyal
- **OnlineSecurity** juga berkontribusi pada retensi
- Investment dalam support services worthwhile

### Statistical Findings

```
Churn Rate Analysis:
├── Overall Churn Rate: 26.7%
├── Tenure < 6 months: ~45% churn
├── Tenure > 60 months: ~5% churn
├── Month-to-month contract: ~43% churn
└── Two-year contract: ~3% churn
```

---

## 📌 Recommendations

### 1. **Focus on New Customer Onboarding**
- Implementasi program welcome intensif di 6 bulan pertama
- Special offers untuk perpanjangan kontrak di bulan ke-3
- Proactive customer support untuk pelanggan baru

### 2. **Promote Long-term Contracts**
- Incentive menarik untuk upgrade ke kontrak tahunan
- Discount khusus untuk two-year contract
- Flexible upgrade options tanpa penalty

### 3. **Improve Fiber Optic Service Quality**
- Investigasi penyebab tingginya churn pada fiber users
- Quality assurance dan network optimization
- Better customer expectation management

### 4. **Implement Targeted Retention Campaign**
- Gunakan model ML untuk scoring pelanggan berisiko
- Automated alerts untuk high-risk customers
- Personalized offers berdasarkan customer profile

### 5. **Strengthen Customer Support**
- Expand TechSupport availability
- Proactive outreach untuk pelanggan tanpa support package
- Bundle support services dengan pricing menarik

---

## 🔮 Future Improvements

### Model Enhancement
- [ ] Ensemble dengan multiple models (stacking)
- [ ] Deep Learning approach (Neural Networks)
- [ ] AutoML untuk automated hyperparameter tuning
- [ ] Real-time prediction API

### Feature Engineering
- [ ] Create interaction features
- [ ] Time-series features (seasonal patterns)
- [ ] Customer lifetime value (CLV) prediction
- [ ] Sentiment analysis dari customer feedback

### Business Integration
- [ ] Integration dengan CRM system
- [ ] Automated email alerts untuk high-risk customers
- [ ] Dashboard untuk monitoring churn trends
- [ ] A/B testing framework untuk retention strategies

### Deployment
- [ ] Deploy ke cloud platform (AWS/GCP/Azure)
- [ ] Dockerize application
- [ ] CI/CD pipeline
- [ ] Model monitoring & retraining automation


---

## 📞 Contact

Jika ada pertanyaan atau feedback, silakan hubungi:

- **Email**: zulfanghifari29@gmail.com
- **LinkedIn**:https://www.linkedin.com/in/zulfanghifari/
- **GitHub**: https://github.com/zulfanghifari


[⬆ Back to Top](#-telco-customer-churn-prediction)

</div>
