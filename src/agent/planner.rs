use anyhow::Result;

use crate::llm::{LlmClient, LlmResponse};

const SYSTEM_PROMPT: &str = r#"You are a research query planner. Given a complex question, decompose it into 1-4 specific search queries that would help find relevant information. Each query should target a different aspect of the question.

Respond with ONLY a JSON array of query strings. Example:
["query 1", "query 2", "query 3"]

Do not include any other text, explanation, or formatting."#;

pub struct Planner {
    llm: LlmClient,
    model: String,
}

impl Planner {
    pub fn new(llm: LlmClient, model: String) -> Self {
        Self { llm, model }
    }

    pub async fn plan(&self, question: &str) -> Result<(Vec<String>, LlmResponse)> {
        let response = self
            .llm
            .complete(&self.model, Some(SYSTEM_PROMPT), question)
            .await?;

        let queries: Vec<String> = serde_json::from_str(&response.text).unwrap_or_else(|_| {
            // Fallback: try to extract JSON array from the response
            if let Some(start) = response.text.find('[') {
                if let Some(end) = response.text.rfind(']') {
                    if let Ok(parsed) = serde_json::from_str(&response.text[start..=end]) {
                        return parsed;
                    }
                }
            }
            // Last resort: use the question itself
            vec![question.to_string()]
        });

        Ok((queries, response))
    }
}
