use image::{DynamicImage, RgbImage, GenericImageView};
use numpy::PyArray3;
use onnxruntime::ndarray::Array1;
use pyo3::prelude::*;

mod face_recognition;
use face_recognition::{Agent, AgentEnvironment};

#[pyclass]
struct PyAgentEnvironment {
    inner: AgentEnvironment,
}

#[pymethods]
impl PyAgentEnvironment {
    #[new]
    fn new() -> eyre::Result<Self> {
        let inner = AgentEnvironment::new()?;
        Ok(Self { inner })
    }

    fn create_agent(&self) -> eyre::Result<PyAgent> {
        let agent = self.inner.create_agent()?;
        Ok(PyAgent { inner: agent })
    }
}

#[pyclass(unsendable)]
struct PyAgent {
    inner: Agent,
}

#[pymethods]
impl PyAgent {
    fn get_embedding(&mut self, image: &PyAny) -> eyre::Result<Vec<f32>> {
        let (h, w, _): (u32, u32, u32) = image.getattr("shape")?.extract()?;
        let input_image = DynamicImage::ImageRgb8(
            RgbImage::from_vec(
                w, h, image.call_method0("flatten")?.extract()?
            ).expect("Failed to convert image to RgbImage")
        );
        let input_image = self.inner.preprocess_input(
            input_image
        )?;
        let embedding = self.inner.run_inference(
            input_image
        )?;
        Ok(embedding.to_vec())
    }

    fn similarity(&mut self, embedding1: Vec<f32>, embedding2: Vec<f32>) -> eyre::Result<f32> {
        Agent::calculate_similarity(
            &Array1::from_shape_vec(512, embedding1)?,
            &Array1::from_shape_vec(512, embedding2)?,
        )
    }

    fn get_similarity(&mut self, image1: Vec<Vec<Vec<u8>>>, image2: Vec<Vec<Vec<u8>>>) -> eyre::Result<f32> {
        let input_image1 = img_from_vec(image1);
        let input_image2 = img_from_vec(image2);
        // Print first pixel value
        self.inner.get_face_similarity(input_image1, input_image2)
    }
}

fn img_from_vec(image: Vec<Vec<Vec<u8>>>) -> DynamicImage {
    let h = image.len() as u32;
    let w = image[0].len() as u32;
    DynamicImage::ImageRgb8(
        RgbImage::from_vec(
            w, h, image.into_iter().flatten().flatten().collect()
        ).expect("Failed to convert image to RgbImage")
    )
}

#[pyfunction]
fn test(a: i32, b: i32) -> PyResult<i32> {
    Ok(a + b)
}

#[pymodule]
fn oxidized_python(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAgentEnvironment>()?;
    m.add_class::<PyAgent>()?;
    m.add_function(wrap_pyfunction!(test, m)?)?;

    Ok(())
}
