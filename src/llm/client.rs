use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct LlmClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

// OpenAI-compatible chat completions format (used by OpenRouter)
#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChatCompletionRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    cost: Option<f64>,
}

// SSE streaming types
#[derive(Debug, Clone, Deserialize)]
struct StreamChatChunk {
    choices: Option<Vec<StreamChatChoice>>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamChatChoice {
    delta: Option<StreamDelta>,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Done {
        full_text: String,
        input_tokens: u32,
        output_tokens: u32,
        cost: f64,
    },
}

impl LlmClient {
    pub fn new(api_key: &str, base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    fn build_messages(system_prompt: Option<&str>, user_message: &str) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_message.to_string(),
        });
        messages
    }

    pub async fn complete(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        user_message: &str,
    ) -> Result<LlmResponse> {
        let messages = Self::build_messages(system_prompt, user_message);

        let request = ChatCompletionRequest {
            model: model.to_string(),
            max_tokens: 4096,
            messages,
        };

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to LLM API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API error ({}): {}", status, body);
        }

        let api_response: ChatCompletionResponse = response
            .json()
            .await
            .context("Failed to parse LLM API response")?;

        let text = api_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        Ok(LlmResponse {
            text,
            input_tokens: api_response.usage.prompt_tokens,
            output_tokens: api_response.usage.completion_tokens,
            cost: api_response.usage.cost.unwrap_or(0.0),
        })
    }

    pub async fn complete_stream(
        &self,
        model: &str,
        system_prompt: Option<&str>,
        user_message: &str,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let messages = Self::build_messages(system_prompt, user_message);

        let mut body = serde_json::to_value(ChatCompletionRequest {
            model: model.to_string(),
            max_tokens: 4096,
            messages,
        })?;
        body.as_object_mut()
            .unwrap()
            .insert("stream".to_string(), serde_json::json!(true));
        // Request usage info in the final streamed chunk
        body.as_object_mut().unwrap().insert(
            "stream_options".to_string(),
            serde_json::json!({"include_usage": true}),
        );

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send streaming request to LLM API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API streaming error ({}): {}", status, body);
        }

        let (tx, rx) = mpsc::channel(256);

        let stream = response.bytes_stream();
        tokio::spawn(async move {
            let mut full_text = String::new();
            let mut input_tokens = 0u32;
            let mut output_tokens = 0u32;
            let mut cost = 0.0f64;
            let mut buffer = String::new();

            tokio::pin!(stream);

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(_) => break,
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete lines from the buffer
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if line == "data: [DONE]" {
                        let _ = tx
                            .send(StreamEvent::Done {
                                full_text: full_text.clone(),
                                input_tokens,
                                output_tokens,
                                cost,
                            })
                            .await;
                        return;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(chunk) = serde_json::from_str::<StreamChatChunk>(data) {
                            // Extract usage from final chunk
                            if let Some(usage) = &chunk.usage {
                                input_tokens = usage.prompt_tokens;
                                output_tokens = usage.completion_tokens;
                                cost = usage.cost.unwrap_or(0.0);
                            }

                            // Extract content delta
                            if let Some(choices) = &chunk.choices {
                                if let Some(choice) = choices.first() {
                                    if let Some(delta) = &choice.delta {
                                        if let Some(content) = &delta.content {
                                            if !content.is_empty() {
                                                full_text.push_str(content);
                                                let _ = tx
                                                    .send(StreamEvent::Token(content.clone()))
                                                    .await;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // If stream ended without [DONE], send Done with what we have
            let _ = tx
                .send(StreamEvent::Done {
                    full_text,
                    input_tokens,
                    output_tokens,
                    cost,
                })
                .await;
        });

        Ok(rx)
    }
}
