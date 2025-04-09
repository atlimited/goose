use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use super::utils::{emit_debug_trace, get_model, handle_response_openai_compat, ImageFormat};
use crate::message::Message;
use crate::model::ModelConfig;
use mcp_core::tool::Tool;

pub const SAMBANOVA_DEFAULT_MODEL: &str = "Meta-Llama-3.1-405B-Instruct";
pub const SAMBANOVA_KNOWN_MODELS: &[&str] = &["Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.3-70B-Instruct"];

pub const SAMBANOVA_DOC_URL: &str = "https://api.sambanova.ai";

#[derive(Debug, serde::Serialize)]
pub struct SambanovaProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    base_path: String,
    api_key: String,
    model: ModelConfig,
    custom_headers: Option<HashMap<String, String>>,
}

impl Default for SambanovaProvider {
    fn default() -> Self {
        let model = ModelConfig::new(SambanovaProvider::metadata().default_model);
        SambanovaProvider::from_env(model).expect("Failed to initialize SambaNova provider")
    }
}

impl SambanovaProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("SAMBANOVA_API_KEY")?;
        let host: String = config
            .get_param("SAMBANOVA_HOST")
            .unwrap_or_else(|_| "https://api.sambanova.ai".to_string());
        let base_path: String = config
            .get_param("SAMBANOVA_BASE_PATH")
            .unwrap_or_else(|_| "v1".to_string());
        let custom_headers: Option<HashMap<String, String>> = config
            .get_secret("SAMBANOVA_CUSTOM_HEADERS")
            .ok()
            .map(parse_custom_headers);
        let timeout_secs: u64 = config.get_param("SAMBANOVA_TIMEOUT").unwrap_or(600);
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()?;

        Ok(Self {
            client,
            host,
            base_path,
            api_key,
            model,
            custom_headers,
        })
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join(&self.base_path).map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let mut request = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        if let Some(custom_headers) = &self.custom_headers {
            for (key, value) in custom_headers {
                request = request.header(key, value);
            }
        }

        let response = request.json(&payload).send().await?;

        handle_response_openai_compat(response).await
    }
}

#[async_trait]
impl Provider for SambanovaProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "sambanova",
            "SambaNova",
            "Meta-Llama-3.1-405B-Instruct model available via SambaNova's API",
            SAMBANOVA_DEFAULT_MODEL,
            SAMBANOVA_KNOWN_MODELS
                .iter()
                .map(|&s| s.to_string())
                .collect(),
            SAMBANOVA_DOC_URL,
            vec![
                ConfigKey::new("SAMBANOVA_API_KEY", true, true, None),
                ConfigKey::new("SAMBANOVA_HOST", true, false, Some("https://api.sambanova.ai")),
                ConfigKey::new("SAMBANOVA_BASE_PATH", true, false, Some("v1")),
                ConfigKey::new("SAMBANOVA_CUSTOM_HEADERS", false, true, None),
                ConfigKey::new("SAMBANOVA_TIMEOUT", false, false, Some("600")),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(&self.model, system, messages, tools, &ImageFormat::OpenAi)?;

        // Make request
        let response = self.post(payload.clone()).await?;

        // Parse response
        let message = response_to_message(response.clone())?;
        let usage = match get_usage(&response) {
            Ok(usage) => usage,
            Err(ProviderError::UsageError(e)) => {
                tracing::debug!("Failed to get usage data: {}", e);
                Usage::default()
            }
            Err(e) => return Err(e),
        };
        let model = get_model(&response);
        emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }
}

fn parse_custom_headers(s: String) -> HashMap<String, String> {
    s.split(',')
        .filter_map(|header| {
            let mut parts = header.splitn(2, '=');
            let key = parts.next().map(|s| s.trim().to_string())?;
            let value = parts.next().map(|s| s.trim().to_string())?;
            Some((key, value))
        })
        .collect()
}
