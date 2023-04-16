use eyre::Result;
use image::{imageops::FilterType::Nearest, DynamicImage};
use onnxruntime::{
    environment::Environment,
    ndarray::{Array1, Array3, Array4, Axis},
    session::Session,
    GraphOptimizationLevel,
};
use std::ops::Deref;
use std::sync::Arc;

const MODEL_PATH: &str = "assets/models/w600k_mbf.onnx";

/// Onnx runtime environment
///
/// This struct is used to initialize the onnxruntime inference environment
/// and create inference agents.
pub struct AgentEnvironment {
    environment: Environment,
}

/// Onnx runtime inference agent
///
/// This struct is used to run inference on a single thread.
pub struct Agent {
    session: Session<'static>,
}

impl AgentEnvironment {
    pub fn new() -> Result<Self> {
        let environment = Self::initialize_environment()?;
        Ok(Self { environment })
    }

    /// Initialize the onnxruntime inference environment
    fn initialize_environment() -> Result<Environment> {
        let env = Environment::builder()
            .with_name("onnxruntime-rs")
            .with_log_level(onnxruntime::LoggingLevel::Verbose)
            .build()?;
        Ok(env)
    }

    /// Create a new inference agent
    pub fn create_agent(&self) -> Result<Agent> {
        let environment = Arc::new(
            self.environment.clone()
        );
        let session = Agent::initialize_session(
            unsafe {
                &*(Arc::as_ptr(&environment) as *const Environment)
            }
        )?;
        Ok(Agent { session })
    }
}

impl Agent {
    /// Initialize inference session using `env` and specified model
    fn initialize_session(env: &'static Environment) -> Result<Session<'static>> {
        let session = env
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(1)?
            .with_model_from_file(MODEL_PATH)?;
        Ok(session)
    }

    /// Preprocess image input. Resize (?, ?, 3) -> (1, 3, input_shape[0], input_shape[1]) and
    /// normalize pixel values to [-1, 1]
    pub fn preprocess_input(&self, image: DynamicImage) -> Result<Vec<Array4<f32>>> {
        let input_dimensions = &self.session.inputs[0].dimensions;
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

    /// Run inference on input tensor
    pub fn run_inference(&mut self, input_tensor: Vec<Array4<f32>>) -> Result<Array1<f32>> {
        let output_tensor = self.session
            .run(input_tensor)?[0]
            .deref()
            .to_owned();
        let output = Array1::<f32>::from_iter(output_tensor.iter().cloned());
        let norm = output.dot(&output).sqrt();
        let output = output / norm;
        Ok(output)
    }

    /// Calculate cosine similarity between two embeddings
    pub fn calculate_similarity(embedding1: &Array1<f32>, embedding2: &Array1<f32>) -> Result<f32> {
        let similarity = embedding1.dot(embedding2);
        Ok(similarity)
    }

    pub fn get_face_similarity(&mut self, image1: DynamicImage, image2: DynamicImage) -> Result<f32> {
        let embedding1 = self.run_inference(self.preprocess_input(image1)?)?;
        let embedding2 = self.run_inference(self.preprocess_input(image2)?)?;
        println!("{:?}", embedding1.get(123));

        let similarity = Self::calculate_similarity(&embedding1, &embedding2)?;
        Ok(similarity)
    }

}
