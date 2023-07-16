import time
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from python.face_embeddings import InferenceBackend
import oxidized_python


class Timer:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.message:
            print(f"{self.message}: {self.interval}")


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


def python_timing():
    root_dir = Path(__file__).parents[0]
    models_dir = root_dir / "assets" / "models"
    detection_model = models_dir / "det_500m.onnx"
    recognition_model = models_dir / "w600k_mbf.onnx"

    face_image_1 = root_dir / "assets" / "tom.png"
    face_image_2 = root_dir / "assets" / "hanks.jpg"
    print(face_image_1)
    face_1 = load_face_image(face_image_1.as_posix())
    face_2 = load_face_image(face_image_2.as_posix())

    Embedder = InferenceBackend(recognition_model.as_posix())


    with Timer("Python inference"):
        for _ in range(10):
            embedding_1 = Embedder(face_1)
            embedding_2 = Embedder(face_2)
            similarity = np.dot(embedding_1, embedding_2)

            del embedding_1, embedding_2
    print("Python similarity:", similarity)


def rust_timing():
    root_dir = Path(__file__).parents[0]
    face_image_1 = root_dir / "assets" / "tom.png"
    face_image_2 = root_dir / "assets" / "hanks.jpg"
    face_1 = load_face_image(face_image_1.as_posix())
    face_2 = load_face_image(face_image_2.as_posix())

    environment = oxidized_python.PyAgentEnvironment()
    agent = environment.create_agent()

    with Timer("Rust inference"):
        for _ in range(10):
            similarity = agent.get_similarity(face_1.tolist(), face_2.tolist())
    print("Rust similarity:", similarity)



if __name__ == "__main__":
    python_timing()
    rust_timing()
