# 🚀 FitPulse — Health Anomaly Detection System

### 📌 Internship Project (Infosys Springboard 6.0 - Python)

---

## 📖 Overview

**FitPulse** is an intelligent health monitoring system designed to analyze fitness data such as heart rate, steps, and sleep patterns to detect anomalies.

The project leverages **data preprocessing, machine learning, and visualization techniques** to identify unusual patterns that may indicate potential health risks.

---

## 🎯 Objective

To build an end-to-end pipeline that:

* Processes raw fitness data
* Extracts meaningful features
* Detects anomalies using multiple techniques
* Visualizes results through an interactive dashboard

---

## 🧰 Tech Stack

* **Programming Language:** Python
* **Libraries Used:**

  * pandas, numpy → Data processing
  * matplotlib → Visualization
  * scikit-learn → Clustering (DBSCAN, KMeans)
  * prophet → Time-series forecasting
  * streamlit → Dashboard UI
* **Version Control:** Git & GitHub

---

## 🏗️ Project Structure

```
FITPULSE/
│── main.py
│── data/
│     └── fitness_data_raw.csv
│── modules/
│     ├── milestone1_preprocessing.py
│     ├── milestone2_modeling.py
│     ├── milestone3_anomaly.py
│     ├── milestone4_dashboard.py
│── outputs/
```

---

## ⚙️ Project Workflow (Milestones)

### 🔹 Milestone 1: Data Collection & Preprocessing

* Loaded raw fitness dataset
* Normalized timestamps
* Handled missing values
* Resampled data into uniform intervals
* Generated cleaned dataset

📌 **Output:**

* `cleaned_data.csv`
* Preprocessing visualization

---

### 🔹 Milestone 2: Feature Extraction & Modeling

* Extracted rolling statistical features
* Applied **Prophet** for time-series forecasting
* Used **KMeans & DBSCAN** for clustering
* Identified potential anomalies

📌 **Output:**

* `feature_matrix.csv`
* Prophet plots
* Clustering visualization

---

### 🔹 Milestone 3: Anomaly Detection

* Applied rule-based anomaly detection
* Combined clustering and statistical anomalies
* Generated anomaly scores and severity levels
* Created visualizations for detected anomalies

📌 **Output:**

* `anomaly_results.csv`
* Heart rate anomaly graph
* Step trend graph
* Daily anomaly summary

---

### 🔹 Milestone 4: Dashboard Visualization

* Built an interactive dashboard using **Streamlit**
* Displayed:

  * Heart rate trends
  * Sleep analysis
  * Step count insights
  * Anomaly summaries

📌 **Outcome:**

* Real-time visualization of health anomalies
* User-friendly interface for analysis

---

## 📊 Key Features

* 🔍 Multi-level anomaly detection
* 📈 Time-series forecasting using Prophet
* 🤖 Machine learning-based clustering
* 📊 Interactive dashboard visualization
* ⚡ Modular and scalable architecture

---

## 🚀 How to Run the Project

### 1️⃣ Run Full Pipeline

```
python main.py
```

### 2️⃣ Launch Dashboard

```
streamlit run modules/milestone4_dashboard.py
```

---

## 📈 Outcome

* Successfully built an **end-to-end anomaly detection system**
* Detected abnormal patterns in health data
* Visualized insights through an interactive dashboard
* Improved understanding of real-world data pipelines and ML integration

---

## 🏁 Conclusion

FitPulse demonstrates how **data science and machine learning** can be applied in healthcare monitoring systems to detect anomalies and provide actionable insights.

This project highlights the integration of:

* Data preprocessing
* Machine learning
* Time-series analysis
* Visualization

into a single cohesive system.

---

## 👨‍💻 Developed As Part Of

**Infosys Springboard 6.0 Internship — Python Track**

---

⭐ If you like this project, consider giving it a star!
