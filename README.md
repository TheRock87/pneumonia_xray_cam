# Pneumonia Detection API

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-username/your-repo/actions)

## Description
This project implements a **Pneumonia Detection API** that classifies chest X-ray images into "NORMAL" or "PNEUMONIA" categories. It leverages a pre-trained DenseNet model and provides visual explanations through **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps, helping to interpret the model's predictions by highlighting important regions in the input image.

The API is built with FastAPI, making it easy to deploy and integrate into other applications. It also includes a Jupyter notebook for model training and experimentation.

### Live Demo
[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo%20on%20Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/hossam87/pneumonia_xray_withCAM_classifier)


## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Features
* **Chest X-ray Image Classification:** Classifies images as "NORMAL" or "PNEUMONIA".
* **Grad-CAM Visualization:** Generates heatmaps to highlight areas of the image that are most indicative of the prediction.
* **FastAPI Backend:** Provides a robust and scalable API for predictions.
* **Pre-trained Deep Learning Model:** Utilizes a DenseNet201 model for high accuracy.
* **Easy Deployment:** Designed for straightforward setup and deployment.

## Technologies Used
* **Languages:** Python
* **Frameworks:** FastAPI
* **Libraries:**
    * `torch` & `torchvision` (for deep learning model)
    * `uvicorn` (ASGI server)
    * `Pillow` (PIL) (for image processing)
    * `scikit-learn` (for utility functions/metrics, if used)
    * `numpy` (for numerical operations)
    * `opencv-python-headless` (for image manipulation, e.g., CAM overlay)
    * `Jinja2` (for templating web pages)
* **Tools:** `pip`

## Folder Structure
```bash
├── app/                                  
│   ├── main.py                           # FastAPI application entry point and API routes.
│   ├── model.py                          # Defines the DenseNet model architecture and loading
│   ├── utils.py                          # Utility functions (e.g., image transformations, Grad-CAM generation, overlay)
│   └── templates/                        # Directory for Jinja2 HTML templates, used for rendering web pages.
│       └── index.html                    # Example: The primary HTML template for the web interface.
├── model/                                
│   └── pneumonia_classifier_weights.pth  # Stores the pre-trained PyTorch model weights.
├── notebook/                            
│   └── pneumonia-detection.ipynb         # Notebook for model training, evaluation, and experimentation.
├── data/                                 # (Optional) Directory for raw or processed datasets, if applicable.
│   └── ...                               
├── requirements.txt                      # Lists all Python package dependencies required for the project.
├── .gitignore                            # Specifies intentionally untracked files and directories to be ignored by Git.
└── README.md                             
```

## Getting Started

Follow these steps to set up and run the Pneumonia Detection API locally.

### Prerequisites
* Python 3.9 or higher
* `pip` (Python package installer)
* `git` (to clone the repository)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Model Weights:**
    Ensure you have the `pneumonia_classifier_weights.pth` file in the `model/` directory. (You might need to train the model using `pneumonia-detection.ipynb` or download pre-trained weights if available from the project maintainer).

5.  **Run the FastAPI application:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to see the interactive API documentation (Swagger UI).

## Usage
The API provides an endpoint to upload a chest X-ray image and get a prediction along with an optional Grad-CAM visualization.

### Web Interface
Access the simple web interface at `http://127.0.0.1:8000/` after starting the server. You can upload an image and view the prediction.

### API Endpoint (`/predict/`)
You can send POST requests to the `/predict/` endpoint with an image file.

**Endpoint:** `POST /predict/`
**Parameters:**
* `file`: The chest X-ray image file (multipart/form-data).
* `generate_cam`: (Optional, boolean) Set to `true` to generate and return a Grad-CAM heatmap overlay. Default is `false`.

**Example using `curl`:**
```bash
curl -X POST "[http://127.0.0.1:8000/predict/?generate_cam=true](http://127.0.0.1:8000/predict/?generate_cam=true)" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg;type=image/jpeg"
```
**Example JSON Response:**
```json
{
  "filename": "your_image.jpg",
  "prediction": "PNEUMONIA",
  "confidence": 98.7654,
  "cam_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..." //Base64 encoded image if generate_cam=true
}
```
