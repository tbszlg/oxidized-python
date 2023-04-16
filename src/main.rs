use eyre::Result;
use image::DynamicImage;
use std::env;
use std::path::Path;

mod face_recognition;
use face_recognition::{Agent, AgentEnvironment};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <path_to_image_1> <path_to_image_2>", args[0]);
        std::process::exit(1);
    }

    let image_path1 = &args[1];
    let image_path2 = &args[2];
    let image1 = load_image(image_path1)?;
    let image2 = load_image(image_path2)?;

    let agent_environment = AgentEnvironment::new()?;
    let mut agent = agent_environment.create_agent()?;

    let input1 = agent.preprocess_input(image1)?;
    let input2 = agent.preprocess_input(image2)?;

    let embedding1 = agent.run_inference(input1)?; // Assuming this method exists
    let embedding2 = agent.run_inference(input2)?; // Assuming this method exists

    let similarity = Agent::calculate_similarity(&embedding1, &embedding2)?;
    println!("Similarity: {similarity}");

    Ok(())
}

fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
    let image = image::open(path)?;
    Ok(image)
}
