use anyhow::Result;
use dotenv::dotenv;
use goose::message::Message;
use goose::model::ModelConfig;
use goose::providers::sambanova::{SambanovaProvider, SAMBANOVA_DEFAULT_MODEL, SAMBANOVA_KNOWN_MODELS};
use goose::providers::base::Provider;
use mcp_core::tool::Tool;
use std::env;

fn load_env() {
    if let Ok(path) = dotenv() {
        println!("Loaded environment from {:?}", path);
    }
}

#[tokio::test]
async fn test_sambanova_model_config() -> Result<()> {
    // Tests that the default model is properly set
    let provider = SambanovaProvider::default();
    assert_eq!(provider.get_model_config().model_name, SAMBANOVA_DEFAULT_MODEL);
    Ok(())
}

#[tokio::test]
async fn test_sambanova_known_models() -> Result<()> {
    // Tests that known models include both Llama models
    assert!(SAMBANOVA_KNOWN_MODELS.contains(&"Meta-Llama-3.1-405B-Instruct"));
    assert!(SAMBANOVA_KNOWN_MODELS.contains(&"Meta-Llama-3.3-70B-Instruct"));
    assert_eq!(SAMBANOVA_KNOWN_MODELS.len(), 2); // Verify we have exactly two models
    Ok(())
}

#[tokio::test]
async fn test_sambanova_custom_model() -> Result<()> {
    // Test creating provider with custom model name
    let model_name = "Meta-Llama-3.3-70B-Instruct";
    let model_config = ModelConfig::new(model_name.to_string());
    
    // Skip if API key not available
    if env::var("SAMBANOVA_API_KEY").is_err() {
        println!("Skipping SambaNova custom model test - API key not set");
        return Ok(());
    }
    
    let provider = SambanovaProvider::from_env(model_config)?;
    assert_eq!(provider.get_model_config().model_name, model_name);
    Ok(())
}

#[tokio::test]
async fn test_sambanova_basic_request() -> Result<()> {
    load_env();
    
    // Skip if API key not available
    if env::var("SAMBANOVA_API_KEY").is_err() {
        println!("Skipping SambaNova basic request test - API key not set");
        return Ok(());
    }
    
    let provider = SambanovaProvider::default();
    let message = Message::user().with_text("Say hello in Japanese");
    
    let (response, usage) = provider
        .complete("You are a helpful assistant.", &[message], &[])
        .await?;
    
    println!("Response: {:?}", response);
    println!("Usage: {:?}", usage);
    
    // Basic validation of response
    assert!(!response.content.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_sambanova_tool_calling() -> Result<()> {
    load_env();
    
    // Skip if API key not available
    if env::var("SAMBANOVA_API_KEY").is_err() {
        println!("Skipping SambaNova tool calling test - API key not set");
        return Ok(());
    }
    
    let provider = SambanovaProvider::default();
    
    let weather_tool = Tool::new(
        "get_weather",
        "Get the weather for a location",
        serde_json::json!({
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            }
        }),
    );
    
    let message = Message::user().with_text("What's the weather like in Tokyo?");
    
    let (response, usage) = provider
        .complete(
            "You are a helpful weather assistant.",
            &[message],
            &[weather_tool],
        )
        .await?;
    
    println!("Tool Response: {:?}", response);
    println!("Usage: {:?}", usage);
    
    // Basic validation that we got some response
    assert!(!response.content.is_empty());
    Ok(())
}
