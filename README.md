# 🩺 Chest X-ray Pneumonia Classifier (TensorFlow + Streamlit)

> **Educational demo** that classifies chest X-rays as **NORMAL** vs **PNEUMONIA** and visualizes model attention with **Grad-CAM**.  
> ⚠️ **Not for clinical use.** Do not use for diagnosis.

---

## ✨ Highlights
- Transfer learning with **MobileNetV2** (ImageNet) in **TensorFlow/Keras**
- **Fine-tuning** for better performance
- **Threshold tuning (τ)** to balance sensitivity & specificity
- **Grad-CAM** overlays for interpretability
- **Streamlit app**: upload an X-ray → prediction + heatmap

---

## 📊 Results (Official Test Set)
- **Accuracy:** ~0.90  
- **ROC AUC:** ~0.965  
- **Recall (PNEUMONIA):** ~0.97  
- **Recall (NORMAL):** ~0.79  
- **Decision threshold (τ):** **0.90** (tuned)

> Metrics computed in `notebooks/01_train_baseline.ipynb` on the dataset’s official test split.

---

## 🧠 Dataset
**Chest X-Ray Images (Pneumonia)** (Kermany et al., hosted on Kaggle)  
Pre-split as `train/`, `val/`, `test/` with two classes: `NORMAL`, `PNEUMONIA`.

Optional Python download (via [kagglehub](https://pypi.org/project/kagglehub/)):
```python
import kagglehub, shutil, pathlib
p = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
src, dst = pathlib.Path(p), pathlib.Path("data")
if not dst.exists():
    shutil.copytree(src, dst)
print("Data available at:", dst)
🚀 Quickstart
bash
Copy
Edit
git clone https://github.com/Huseen1141/medical-xray-classifier.git
cd medical-xray-classifier

# (Windows PowerShell) create & activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# install
pip install -r requirements.txt

# run the demo app
streamlit run app/streamlit_app.py
📓 Train • Tune • Evaluate (Notebook)
Everything is in notebooks/01_train_baseline.ipynb:

Load & preprocess (RGB, (224,224))

Baseline training (MobileNetV2 backbone frozen)

Fine-tune the top backbone layers

Evaluate on test set (accuracy, AUC, precision/recall/F1, confusion matrix)

Tune threshold (τ) on a larger validation split (e.g., 15% of train)

Save artifacts:

models/mobilenetv2_finetuned.keras (ignored by git)

models/class_names.json

models/threshold.json (used by the app)

Grad-CAM overlay to visualize model focus

The Streamlit app loads class_names.json and threshold.json. The .keras model stays local (to keep the repo light).
🧰 Tech stack
TensorFlow / Keras (MobileNetV2, transfer learning, fine-tuning)

scikit-learn (metrics, threshold tuning)

Streamlit (interactive app UI)

OpenCV, Pillow, Matplotlib (image I/O & visualization)

Jupyter / VS Code (experiments, reproducibility)

Git & GitHub (version control)

Key versions pinned in requirements.txt (Windows-friendly):
tensorflow==2.16.1, numpy<2.0, opencv-python<4.11, plus Streamlit, sklearn, etc.

 Limitations & Ethics
Trained on a public research dataset; not clinically validated.

Performance can vary with population, device, and protocol shift.

Use only for learning and prototyping — not for medical decisions.

 Project structure
bash
Copy
Edit
medical-xray-classifier/
 ├─ app/                    # Streamlit app (upload + Grad-CAM + τ slider)
 ├─ data/                   # dataset (ignored by git)
 ├─ models/                 # model (.keras ignored), class_names.json, threshold.json
 ├─ notebooks/              # 01_train_baseline.ipynb (train/tune/eval/Grad-CAM)
 ├─ src/                    # helper scripts (e.g., download/check)
 ├─ docs/assets/            # screenshots for README (optional)
 ├─ requirements.txt
 └─ README.md
  Troubleshooting
PowerShell blocks venv activation:
Run as user:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Model file not found in app:
Re-run notebook Cell 4 to save the model and JSONs, launch Streamlit from the project root.

OpenCV / NumPy mismatch:
Keep numpy<2.0 and use opencv-python<4.11 (already pinned).
References
Kermany, D. S., et al. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification.

Kaggle dataset: Chest X-Ray Images (Pneumonia) — paultimothymooney/chest-xray-pneumonia

Selvaraju, R. R., et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

