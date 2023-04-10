use image::open;
use pyo3::prelude::*;

mod face_recognition;

const IMAGE_PATH_TOM: &str = "assets/tom.png";
const IMAGE_PATH_HANKS: &str = "assets/hanks.jpg";

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn get_face_embeddings() -> PyResult<f32> {
    let env = face_recognition::initialize_environment().unwrap();
    let mut session = face_recognition::initialize_session(&env).unwrap();
    let image = open(IMAGE_PATH_TOM).unwrap();
    let input = face_recognition::preprocess_input(image, &session).unwrap();
    let output_1 = face_recognition::run_inference(input, &mut session).unwrap();

    let image = open(IMAGE_PATH_HANKS).unwrap();
    let input = face_recognition::preprocess_input(image, &session).unwrap();
    let output_2 = face_recognition::run_inference(input, &mut session).unwrap();

    let similarity = face_recognition::calculate_similarity(&output_1, &output_2).unwrap();
    Ok(similarity)
}

/// A Python module implemented in Rust.
#[pymodule]
fn oxidized_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(get_face_embeddings, m)?)?;
    Ok(())
}
