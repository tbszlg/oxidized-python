use eyre::Result;
use image::{imageops::FilterType::Nearest, DynamicImage};
use onnxruntime::{
    environment::Environment,
    ndarray::{Array1, Array3, Array4, Axis},
    session::Session,
    GraphOptimizationLevel,
};
use std::ops::Deref;

const MODEL_PATH: &str = "assets/models/w600k_mbf.onnx";


/// Initialize the onnxruntime inference environment
pub fn initialize_environment() -> Result<Environment> {
    let env = Environment::builder()
        .with_name("onnxruntime-rs")
        .with_log_level(onnxruntime::LoggingLevel::Verbose)
        .build()?;
    Ok(env)
}


/// Initialize inference session using `env` and specified model
pub fn initialize_session(env: &Environment) -> Result<Session> {
    let session = env
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file(MODEL_PATH)?;
    Ok(session)
}


/// Preprocess image input. Resize (?, ?, 3) -> (1, 3, input_shape[0], input_shape[1]) and
/// normalize pixel values to [-1, 1]
pub fn preprocess_input(image: DynamicImage, session: &Session) -> Result<Vec<Array4<f32>>> {
    let input_dimensions = &session.inputs[0].dimensions;
    let image_resized = image
        .resize_exact(input_dimensions[2].unwrap(), input_dimensions[3].unwrap(), Nearest)
        .to_rgb8();
    let height = image_resized.height();
    let width = image_resized.width();
    let image_resized = Array3::from_shape_fn((height as usize, width as usize, 3 as usize), |(i, j, k)| {
        image_resized[(j as u32, i as u32)][k]
    })
        .permuted_axes([2, 0, 1]);

    let input_image = image_resized
        .insert_axis(Axis(0))
        .map(|v| *v as f32 / 127.5 - 1.0);
    let input_tensor = vec![input_image];
    Ok(input_tensor)
}

pub fn run_inference(input_tensor: Vec<Array4<f32>>, session: &mut Session) -> Result<Array1<f32>> {
    let output_tensor = session
        .run(input_tensor)?[0]
        .deref()
        .to_owned();
    let output = Array1::<f32>::from_iter(output_tensor.iter().cloned());
    let norm = output.dot(&output).sqrt();
    let output = output / norm;
    Ok(output)
}


pub fn calculate_similarity(embedding1: &Array1<f32>, embedding2: &Array1<f32>) -> Result<f32> {
    let similarity = embedding1.dot(embedding2);
    Ok(similarity)
}
