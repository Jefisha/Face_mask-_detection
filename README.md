# Face Mask Detection with OpenCV and Keras

This project uses OpenCV and a trained Keras model to detect whether a person is wearing a face mask in real-time using your webcam.

## Files

- `face_mask_model.h5`: Pre-trained Keras model for mask detection.
- [`test.py`](test.py): Python script for running real-time mask detection.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- TensorFlow (with Keras)

Install dependencies with:

```sh
pip install opencv-python numpy tensorflow
```

## Usage

1. Place `face_mask_model.h5` in the project directory.
2. Run the detection script:

   ```sh
   python test.py
   ```

3. The webcam will open and display a window showing mask detection results. Press `q` to quit.

## How it works

- Detects faces using Haar cascades.
- Crops and preprocesses the largest detected face.
- Uses the Keras model to predict "Mask" or "No Mask".
- Displays the result and draws a colored rectangle around the face.

## License

This project is for educational purposes.
