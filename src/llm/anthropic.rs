use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct AnthropicClient {
    client: reqwest::Client,
    api_key: String,
}

#[derive(Debug, Clone, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    system: Option<String>,
    messages: Vec<Message>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiResponse {
    pub content: Vec<ContentBlock>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl AnthropicClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
        }
    }

    pub async fn complete(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        user_message: &str,
    ) -> Result<LlmResponse> {
        let request = ApiRequest {
            model: model.to_string(),
            max_tokens: 4096,
            system: system_prompt.map(|s| s.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: user_message.to_string(),
            }],
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error ({}): {}", status, body);
        }

        let api_response: ApiResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic API response")?;

        let text = api_response
            .content
            .iter()
            .filter_map(|block| block.text.as_deref())
            .collect::<Vec<_>>()
            .join("");

        Ok(LlmResponse {
            text,
            input_tokens: api_response.usage.input_tokens,
            output_tokens: api_response.usage.output_tokens,
        })
    }
}
