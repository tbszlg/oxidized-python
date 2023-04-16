use image::{DynamicImage, RgbImage};
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
    fn new() -> PyResult<Self> {
        let inner = AgentEnvironment::new().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(Self { inner })
    }

    fn create_agent(&self) -> PyResult<PyAgent> {
        let agent = self.inner.create_agent().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
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
