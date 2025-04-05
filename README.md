# Video Surveillance - Fight Detection Algorithm

This project implements a **fight detection algorithm** using a combination of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks. The algorithm is designed to classify video frames into two categories: **Violence** and **No Violence**.

## Core Details

### 1. **Dataset and Preprocessing**
- **Dataset**: The project uses the "Real Life Violence Dataset" containing videos labeled as either violent or non-violent.
- **Frame Extraction**: 
    - Each video is processed to extract frames using OpenCV.
    - Frames are resized to `224x224` pixels and normalized to the range `[0, 1]`.
    - A maximum of 20 frames per video is used for training.
- **Labels**:
    - Videos starting with `V` are labeled as `[1, 0]` (Violence).
    - Videos starting with `NV` are labeled as `[0, 1]` (No Violence).
    - Labels and video paths are shuffled for randomness.

### 2. **Feature Extraction with Pre-trained VGG16**
- **VGG16 Model**:
    - A pre-trained VGG16 model is used for feature extraction.
    - The `fc2` layer of VGG16 is used as the transfer layer to extract high-level features from video frames.
- **Transfer Values**:
    - Transfer values are computed for each frame and stored in `.h5` files for efficient processing.

### 3. **Data Preparation**
- **Training and Testing Split**:
    - 80% of the dataset is used for training, and 20% is used for testing.
- **HDF5 Storage**:
    - Training and testing data are stored in HDF5 files (`prueba.h5` and `pruebavalidation.h5`) for efficient loading during training.

### 4. **Model Architecture**
- **LSTM-based Model**:
    - Input: Sequence of transfer values (20 frames per video, each with 4096 features).
    - Layers:
        - LSTM layer with 512 units.
        - Dense layer with 1024 units and ReLU activation.
        - Dense layer with 50 units and sigmoid activation.
        - Output layer with 2 units and softmax activation for binary classification.
    - Loss Function: Mean Squared Error.
    - Optimizer: Adam.
    - Metrics: Accuracy.

### 5. **Training**
- **Hyperparameters**:
    - Epochs: 200
    - Batch Size: 500
- **Validation**:
    - 10% of the training data is used for validation during training.
- **History**:
    - Training and validation accuracy and loss are plotted for analysis.

### 6. **Evaluation**
- The model is evaluated on the test dataset, and metrics such as accuracy and loss are reported.
- Plots for accuracy and loss over epochs are saved in `.eps` format.

### 7. **Model Saving**
- The trained model is saved in the `models/` directory as `saved_model_v5`.

### 8. **Utilities**
- **Progress Tracking**: A utility function `print_progress` is used to display the progress of data processing.
- **Data Processing**: Functions like `extract_frames`, `create_labels`, and `process_alldata_training_and_test` handle data preparation and processing.

### 9. **Dependencies**
- Python libraries used:
    - `cv2` (OpenCV) for video processing.
    - `tensorflow` and `keras` for deep learning.
    - `numpy`, `pandas`, and `matplotlib` for data manipulation and visualization.
    - `h5py` for HDF5 file handling.

### 10. **Output**
- The trained model can classify video sequences into violent or non-violent categories.
- Accuracy and loss plots provide insights into the model's performance.

### 11. **File Structure**
- **`fight_detection_algo.py`**: Main script containing the implementation.
- **`dataset/`**: Directory containing the video dataset.
- **`models/`**: Directory to save the trained model.
- **`prueba.h5` and `pruebavalidation.h5`**: HDF5 files storing processed training and testing data.

### 12. **Future Improvements**
- Use a larger dataset for better generalization.
- Experiment with different architectures and hyperparameters.
- Implement real-time video processing for live surveillance.

### 13. **API Server**

The project includes an API server built using **FastAPI** to enable video upload and processing for fight detection. Below are the details of the API server:

#### **Endpoints**

1. **`GET /`** - Home Page
    - **Description**: Serves the home page of the application.
    - **Response**: Renders an HTML template (`home.html`) with a form for uploading video files.
    - **Parameters**: None.

2. **`POST /submitform`** - Video Upload and Processing
    - **Description**: Accepts video files uploaded by the user, processes them to detect violent scenes, and returns a downloadable ZIP file containing the processed video clips.
    - **Parameters**:
      - `uploaded_files`: A list of video files (only `.mp4` format is supported).
    - **Response**:
      - Returns a ZIP file containing video clips with detected violent scenes.
      - If no violent scenes are detected, the ZIP file will be empty.

#### **Processing Workflow**

1. **Video Upload**:
    - Uploaded video files are saved in the `received/` directory on the server.

2. **Violence Detection**:
    - Each uploaded video is processed using the `detect` function.
    - The function extracts frames from the video, resizes them to `224x224` pixels, and computes transfer values using the pre-trained VGG16 model.
    - Frames are grouped into batches of 20, and predictions are made using the trained LSTM-based model.
    - If a batch is classified as "Violence," the corresponding frames are saved as a new video clip.

3. **Output Generation**:
    - Detected violent scenes are saved as separate video files in the `processed_files/` directory.
    - All processed video files are compressed into a single ZIP file for download.
    - The original uploaded video is deleted after processing.

4. **Error Handling**:
    - The server ensures that only `.mp4` files are processed.
    - Directories (`received/` and `processed_files/`) are created automatically if they do not exist.

#### **Directory Structure**
- **`received/`**: Stores uploaded video files temporarily.
- **`processed_files/`**: Stores processed video clips and ZIP files.

#### **Technologies Used**
- **FastAPI**: For building the API server.
- **Jinja2**: For rendering HTML templates.
- **OpenCV**: For video frame extraction and processing.
- **TensorFlow/Keras**: For loading the pre-trained model and making predictions.
- **Concurrent Futures**: For parallel processing of video frames.
- **ZipFile**: For compressing processed video clips into a ZIP file.

#### **How to Run the API Server**
1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Start the server:
    ```bash
    uvicorn main:app --reload
    ```
3. Access the home page at `http://127.0.0.1:8000/`.

#### **Future Enhancements**
- Add support for additional video formats (e.g., `.avi`, `.mov`).
- Implement real-time video streaming and processing.
- Enhance the user interface for better usability.
- Add authentication and authorization for secure access.
