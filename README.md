# Potato Disease Classification

This project implements a FastAPI-based web application for classifying potato plant diseases using deep learning. The application can identify three conditions:
- Potato Early Blight
- Potato Late Blight
- Healthy Potato Plant

## Features
- Web interface for image upload
- Real-time disease classification
- REST API endpoint for predictions
- Docker support for easy deployment

## Setup and Installation

### Using Docker (Recommended)
1. Clone the repository:
```bash
git clone https://github.com/Atasatti/Potato_disease_Classification.git
cd Potato_disease_Classification
```

2. Build and run using Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`

### Manual Setup
1. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints
- `GET /`: Web interface
- `POST /predict/`: Upload an image for disease classification

## Technologies Used
- FastAPI
- TensorFlow/Keras
- OpenCV
- Docker
- Python 3.10 