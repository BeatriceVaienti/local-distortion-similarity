# Local Distortion Similarity Analysis

Code from the paper Segmentation and **"Clustering of Local Planimetric Distortion Patterns in Historical Maps of Jerusalem"**

This repository contains code for analyzing **local distortions in historical maps**, using **geospatial clustering, similarity computation, and transformation functions**. The project is structured into three main **modules**: `clustering`, `comparison`, and `transform`, with testing done in `test-local-distortion-similarity.ipynb`.

## 📂 Project Structure
```
local-distortion-similarity/
│── modules/
│   ├── clustering.py      # Functions for clustering regions
│   ├── comparison.py      # Functions for computing similarity & metrics
│   ├── transform.py       # CRS transformations and coordinate conversions
│── test-local-distortion-similarity.ipynb  # Jupyter Notebook for testing
│── README.md              # Project documentation
│── LICENSE                # License details
│── maps/                  # Sample maps for testing
```

---

## 🔍 **Modules Overview**



###  **1. `comparison.py`**
Performs **geometric similarity analysis** for map distortion evaluation.

---

###  **2. `clustering.py`**
Handles **spatial clustering of distorted regions**.

---

###  **3. `transform.py`**
Handles **coordinate system transformations** between different EPSG codes.
NB: Remember to correct the EPSG codes to match the desired coordinate system.

---

## 📊 **Testing & Usage**
Testing is performed in the Jupyter Notebook:
- **`test-local-distortion-similarity.ipynb`** contains **example runs & validation**.

---

## 🔗 **License**
This project is licensed under **MIT License**. See `LICENSE` for details.
