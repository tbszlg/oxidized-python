import typing as t
from pathlib import Path

import cv2
import numpy as np
import onnxruntime


def load_face_image(image_path: str) -> np.ndarray:
    face = cv2.imread(image_path)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face


class InferenceBackend:
    def __init__(self, model_file_path: str):
        self.model_file_path = model_file_path

        self.input_name: str
        self.input_shape: t.Tuple

        self.session: onnxruntime.InferenceSession

        self._setup_onnx()
        self.processor = InferenceProcessor(input_shape=self.input_shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run inference on input data `x`.

        Depending on the backend that was determined based on the environment
        architecture the inference is run using onnx or TensorRT.

        Args:
            x (np.ndarray): Input image.

        Returns:
            np.ndarray: Inference result.
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
        if self.input_shape == ["?", "?"]:
            self.input_shape = (640, 640)

    def _run_onnx(self, x: np.ndarray) -> np.ndarray:
        """Run inference using the onnx runtime session."""
        return self.session.run(None, {self.input_name: x})


class InferenceProcessor:
    """Pre- and post-processor for model inference."""

    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Preprocess input data for inference."""
        x = cv2.resize(x, (self.input_shape[0], self.input_shape[1]))
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)

        # Normalize to -1/1 range
        x = x / 255.0
        x = (x - 0.5) / 0.5

        return x

    def postprocess(self, y: np.ndarray) -> np.ndarray:
        """Postprocess inference results."""
        y = y[0].flatten()
        y /= np.linalg.norm(y)
        return y


if __name__ == "__main__":
    root_dir = Path(__file__).parents[1]
    models_dir = root_dir / "assets" / "models"
    detection_model = models_dir / "det_500m.onnx"
    recognition_model = models_dir / "w600k_mbf.onnx"

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
