use anyhow::Result;

use crate::llm::{LlmClient, LlmResponse};

const SYSTEM_PROMPT: &str = r#"You are a research synthesizer. Given a question and accumulated research context (passages retrieved across multiple search hops), provide a comprehensive, well-structured answer.

Guidelines:
- Synthesize information from multiple passages into a coherent answer
- Note connections between different pieces of information
- Be specific — cite facts from the passages rather than making general statements
- If the evidence is insufficient or contradictory, say so
- Keep the answer focused and concise (2-4 paragraphs)"#;

pub struct Synthesizer {
    llm: LlmClient,
    model: String,
}

impl Synthesizer {
    pub fn new(llm: LlmClient, model: String) -> Self {
        Self { llm, model }
    }

    pub async fn synthesize(
        &self,
        question: &str,
        accumulated_context: &[String],
    ) -> Result<(String, LlmResponse)> {
        let context_text = accumulated_context
            .iter()
            .enumerate()
            .map(|(i, p)| format!("[Source {}] {}", i + 1, p))
            .collect::<Vec<_>>()
            .join("\n\n");

        let user_message = format!(
            "Question: {}\n\nResearch Context:\n{}",
            question, context_text
        );

        let response = self
            .llm
            .complete(&self.model, Some(SYSTEM_PROMPT), &user_message)
            .await?;

        Ok((response.text.clone(), response))
    }
}
