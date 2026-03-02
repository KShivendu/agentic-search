use anyhow::Result;
use tokio::sync::mpsc;

use crate::llm::{LlmClient, LlmResponse, StreamEvent};

const SYSTEM_PROMPT: &str = r#"You are a research synthesizer. Given a question and research context (numbered passages retrieved via search), provide a comprehensive, well-structured answer grounded strictly in the provided sources.

Rules:
- ONLY use information that appears in the provided sources. Do NOT add outside knowledge.
- Cite every factual claim with inline source references like [1], [2], or [1][3] for multiple sources.
- If the sources do not contain enough information to fully answer the question, explicitly state what is missing rather than guessing.
- If sources contradict each other, note the contradiction and cite both sides.
- Synthesize across sources into a coherent answer — do not just list source summaries.
- Keep the answer focused and concise (2-4 paragraphs)."#;

pub struct Synthesizer {
    llm: LlmClient,
    model: String,
}

impl Synthesizer {
    pub fn new(llm: LlmClient, model: String) -> Self {
        Self { llm, model }
    }

    fn build_user_message(question: &str, accumulated_context: &[String]) -> String {
        let context_text = accumulated_context
            .iter()
            .enumerate()
            .map(|(i, p)| format!("[Source {}] {}", i + 1, p))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            "Question: {}\n\nResearch Context:\n{}",
            question, context_text
        )
    }

    pub async fn synthesize(
        &self,
        question: &str,
        accumulated_context: &[String],
    ) -> Result<(String, LlmResponse)> {
        let user_message = Self::build_user_message(question, accumulated_context);

        let response = self
            .llm
            .complete(&self.model, Some(SYSTEM_PROMPT), &user_message)
            .await?;

        Ok((response.text.clone(), response))
    }

    pub async fn synthesize_stream(
        &self,
        question: &str,
        accumulated_context: &[String],
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let user_message = Self::build_user_message(question, accumulated_context);

        self.llm
            .complete_stream(&self.model, Some(SYSTEM_PROMPT), &user_message)
            .await
    }
}
