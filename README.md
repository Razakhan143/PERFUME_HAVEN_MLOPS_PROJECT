# 🌸 PERFUME HAVEN - MLOps Project

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MLOps](https://img.shields.io/badge/MLOps-DVC%20%2B%20Docker-blueviolet.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Supported-326CE5.svg)]()

*An elegant ML pipeline orchestration system for perfume recommendation and analysis*

[Features](#features) • [Quick Start](#-quick-start) • [Dashboard](#-dashboard-showcase) • [Architecture](#-architecture) • [Tech Stack](#-comprehensive-tech-stack) • [Running](#-how-to-run) • [Docs](#-documentation)

</div>

---

## 📋 Overview

**Perfume Haven** is a production-ready MLOps project demonstrating best practices in machine learning pipeline development, containerization, and deployment. The system analyzes perfume characteristics, predicts customer preferences, and provides personalized recommendations through an interactive dashboard using cosine similarity-based search recommendation system.

**Key Highlights:**
- 🎯 End-to-end ML pipeline with 6 automated stages
- 📊 Interactive Plotly dashboards with real-time analytics
- 🐳 Docker containerized with production-grade deployment
- 🔄 DVC versioning for reproducible ML workflows
- ⚙️ Kubernetes ready for scalable deployments
- 📈 MLflow experiment tracking and model registry
- ���� Production-grade monitoring and logging

---

## ✨ Features

- **🎯 End-to-End ML Pipeline**: Data ingestion → preprocessing → feature engineering → model training → evaluation → registration
- **📊 Interactive Dashboards**: Real-time analytics and perfume recommendations visualization with Plotly
- **🐳 Docker Containerization**: Seamless deployment with Gunicorn + Uvicorn workers
- **🔄 DVC Integration**: Complete ML reproducibility with data and model versioning
- **⚙️ Kubernetes Support**: Production-grade Kubernetes deployment configuration
- **📈 Model Monitoring**: MLflow experiment tracking, Prometheus metrics collection
- **🔐 AWS Integration**: S3 storage, IAM authentication, cloud-native deployment
- **🧪 Comprehensive Testing**: Unit tests and environment validation
- **📝 Production-Ready Code**: Following MLOps best practices and industry standards

---







## 📊 Project Showcase

<table>
<tr>
<td width="50%">

### 📈 Main Analytics Dashboard
![Analytics Dashboard](https://github.com/user-attachments/assets/458aaf10-1bf2-4c25-b749-fd354de40337)
*Real-time perfume analytics and KPI tracking*

</td>
<td width="50%">

### 🎁 Perfume Recommendations Engine
![Recommendations](https://github.com/user-attachments/assets/2dd591cc-5615-4aed-8f4b-bfeccdb99cd8)
*AI-powered personalized recommendations*

</td>
</tr>
<tr>
<td width="50%">

### CI/CD
![Customer Analytics](https://github.com/user-attachments/assets/249774bb-5b65-453b-89af-d05084fc74b1)
*Continous Integration and Continous Development till deployment to Azure*

</td>
<td width="50%">

### Amazon Web Services 
![Model Performance](https://github.com/user-attachments/assets/a0f330ac-d1cc-47ae-aed3-23875b8bbeec)
*Amazon EC2 Instance to Deploy Backend in Kubernetes*

</td>
</tr>
<tr>
<td width="50%">

### MLflow Panle
![Trend Analysis](https://github.com/user-attachments/assets/f391ce95-4fe8-43f3-9147-c49a0c879639)
*Experiment Tracking for ML models*

</td>
<td width="50%">

### prometheous & Graphana
![Monitoring System](https://github.com/user-attachments/assets/0e9d8c41-9f4e-450f-9a9c-6ecc8abe391f)
*to monitor the performance of the Application and the model*

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (recommended)
- **Docker** 20.10+ and **Docker Compose** 1.29+
- **Git** for version control
- **Make** (optional, for convenient commands)
- **kubectl** (optional, for Kubernetes deployment)

### Installation Options

#### Option 1: Local Installation (Development)

```bash
# Clone the Repository
git clone https://github.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.git
cd PERFUME_HAVEN_MLOPS_PROJECT

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Verify Environment
python test_environment.py
