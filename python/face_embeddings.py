import typing as t
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime


class InferenceBackend:
    """Inference backend for face recognition.

    The backend is responsible for setting up the ONNX environment
    for inference and running the inference.
    """

    def __init__(self, model_file_path: str):
        """Initialize the inference backend.

        Args:
            model_file_path (str): Path to the ONNX model file.
        """
        self.model_file_path: str = model_file_path

        self.input_name: str
        self.input_shape: t.Tuple[int, int]

        self.session: onnxruntime.InferenceSession

        self._setup_onnx()
        self.processor: InferenceProcessor = InferenceProcessor(input_shape=self.input_shape)

    def __call__(self, x: npt.NDArray) -> npt.NDArray[np.float32]:
        """Run inference on input data `x`.

        Depending on the backend that was determined based on the environment
        architecture the inference is run using onnx or TensorRT.

        Args:
            x (NDArray): Input image.

        Returns:
            NDArray[float32]: Inference result.
        """
        x = self.processor.preprocess(x)
        y = self._run_onnx(x)
        y = self.processor.postprocess(y)
        return y

    def _setup_onnx(self) -> None:
        """Create the onnx inference runtime."""
        self.session = onnxruntime.InferenceSession(self.model_file_path)
        input = self.session.get_inputs()[0]
        self.input_name = input.name
        self.input_shape = input.shape[2:]

        # If the model does not specify the input shape
        # we assume it is 640x640
        if self.input_shape == ["?", "?"]:
            self.input_shape = (640, 640)

    def _run_onnx(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run inference using the onnx runtime session.

        Args:
            x (NDArray[float32]): Preprocessed input image.

        Returns:
            NDArray[float32]: Inference result.
        """
        return self.session.run(None, {self.input_name: x})


class InferenceProcessor:
    """Pre- and post-processor for model inference."""

    def __init__(self, input_shape: t.Tuple[int, int]):
        self.input_shape = input_shape

    def preprocess(self, x: npt.NDArray) -> npt.NDArray[np.float32]:
        """Preprocess input data for inference.

        The preprocessing steps are:
        - Resize to model input shape
        - Transpose to NCHW
        - Add batch dimension
        - Convert to float32
        - Normalize to -1/1 range

        Args:
            x (NDArray): Input image.

        Returns:
            NDArray[float32]: Preprocessed input image.
        """
        x = cv2.resize(x, (self.input_shape[0], self.input_shape[1]))
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)

        # Normalize to -1/1 range
        x = x / 255.0
        x = (x - 0.5) / 0.5

        return x

    def postprocess(self, y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Postprocess inference results.

        The postprocessing steps are:
        - Flatten the output
        - Normalize to unit length

        Args:
            y (NDArray[float32]): Inference result.

        Returns:
            NDArray[float32]: Postprocessed inference result.
        """
        y = y[0].flatten()
        y /= np.linalg.norm(y)
        return y


if __name__ == "__main__":

    def load_face_image(image_path: str) -> npt.NDArray[np.uint8]:
        """Helper function to load face images from disk.

        Args:
            image_path (str): Path to the image file.

        Returns:
            NDArray[uint8]: Face image.
        """
        face = cv2.imread(image_path)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.uint8)
        return face

    # Define model paths
    root_dir = Path(__file__).parents[1]
    models_dir = root_dir / "assets" / "models"
    detection_model = models_dir / "det_500m.onnx"
    recognition_model = models_dir / "w600k_mbf.onnx"

    # Initialize ONNX inference backend
    Embedder = InferenceBackend(recognition_model.as_posix())

    # Run inference on dummy data
    face_image_1 = root_dir / "assets" / "tom.png"
    face_image_2 = root_dir / "assets" / "hanks.jpg"
    face_1 = load_face_image(face_image_1.as_posix())
    face_2 = load_face_image(face_image_2.as_posix())
    embedding_1 = Embedder(face_1)
    embedding_2 = Embedder(face_2)
    similarity = np.dot(embedding_1, embedding_2)

    print(f"Similarity: {similarity}")
