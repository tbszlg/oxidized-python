use image::{RgbImage, DynamicImage};
use onnxruntime::ndarray::Array1;
use pyo3::prelude::*;

mod face_recognition;


#[pyfunction]
fn embedding(image: &PyAny) -> eyre::Result<Vec<f32>> {
    let env = face_recognition::initialize_environment()?;
    let mut session = face_recognition::initialize_session(&env)?;

    let (h, w, _): (u32, u32, u32) = image.getattr("shape")?.extract()?;
    let input_img = DynamicImage::ImageRgb8(
        RgbImage::from_vec(
            w, h, image.call_method0("flatten")?.extract()?
        ).expect("Failed to convert to RgbImage")
    );
    let input_img = face_recognition::preprocess_input(input_img, &session)?;
    let embedding = face_recognition::run_inference(input_img, &mut session)?;
    Ok(embedding.to_vec())
}


#[pyfunction]
fn similarity(embedding1: Vec<f32>, embedding2: Vec<f32>) -> eyre::Result<f32> {
    face_recognition::calculate_similarity(
        &Array1::<f32>::from_shape_vec(512, embedding1)?,
        &Array1::<f32>::from_shape_vec(512, embedding2)?,
    )
}


/// A Python module implemented in Rust.
#[pymodule]
fn oxidized_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embedding, m)?)?;
    m.add_function(wrap_pyfunction!(similarity, m)?)?;
    Ok(())
}
