# Brain_Tumor_Detection_and_Survival_Prediction

# 🧠 Brain Tumor Detection and Survival Prediction App

This project is a Streamlit-based web application that allows users to:
- Upload **3D MRI scans** in NIfTI format (`.nii` / `.nii.gz`)
- Automatically **segment brain tumors** using a pretrained 3D U-Net model
- Predict **patient survival likelihood** based on tumor characteristics
- View **multi-slice visualizations**, SHAP explanations, tumor volume, and more
- Download and optionally email a **PDF report**

---

## 📂 Project Structure

Brain\_Tumor\_Detection\_and\_Survival\_Prediction/
├── appy.py                     # Main Streamlit app
├── brain_tumor_3D-Unet.ipynb   # Model development notebook
├── preprocess.py               # Preprocessing utilities
├── survival.py                 # Survival prediction logic
├── heathy.py                   # Healthy image utilities
├── 3d_unet_brain_tumor_segmenter.pth  # Pretrained segmentation model
├── rf_survival_model.pkl       # Trained Random Forest model
├── survival_info.csv           # Sample data used for training/testing
├── requirements.txt            # Python dependencies
└── README.md                   # This file

---

## 🚀 Features

- 📤 Upload NIfTI MRI volumes
- 🧠 Tumor segmentation with **3D U-Net (MONAI)**
- 📊 Predict survival using **Random Forest**
- 📈 Interactive SHAP feature explanations
- 📸 Animated multi-slice tumor GIF viewer
- 📄 In-memory PDF report generation
- 📧 Optional email report delivery

---

## 🔧 Installation

### 1. Clone the repository

git clone https://github.com/Phoenix-Krn/Brain_Tumor_Detection_and_Survival_Prediction.git
cd Brain_Tumor_Detection_and_Survival_Prediction

### 2. Create a virtual environment (recommended)

python -m venv env
source env/bin/activate       # On Linux/macOS
env\Scripts\activate          # On Windows

### 3. Install dependencies

pip install -r requirements.txt

---

## ▶️ Run the App Locally

streamlit run appy.py

Then open the local URL it gives (usually `http://localhost:8501`) in your browser.

---

## 📥 Sample Files & Model

* `3d_unet_brain_tumor_segmenter.pth`: 3D U-Net tumor segmentation model
* `rf_survival_model.pkl`: Trained Random Forest for survival prediction

If models are too large for GitHub:

* Upload to Google Drive
* Share public link
* Load with `gdown` in code

---

## 🧪 Example Use Cases

* Medical students learning tumor detection
* AI research in healthcare
* Fast clinical prototypes for radiology support tools

---

## 🧠 Authors

* **Phoenix-Krn** – GitHub: [@Phoenix-Krn](https://github.com/Phoenix-Krn)

---

## 📜 License

This project is for **educational and research purposes only**. Not intended for clinical use.

---

## 🙋‍♀️ Questions?

Open an issue or contact via GitHub.

---
