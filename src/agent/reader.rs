use anyhow::Result;

use crate::llm::{AnthropicClient, LlmResponse};

const SYSTEM_PROMPT: &str = r#"You are a research reader. You are given a question, retrieved passages, and context accumulated from previous research hops.

Your job is to decide:
1. If you have enough information to answer the question, respond with:
   {"decision": "synthesize"}

2. If you need more information, respond with:
   {"decision": "continue", "follow_up_queries": ["query 1", "query 2"]}
   Provide 1-3 follow-up queries targeting specific gaps in your knowledge.

Consider:
- What aspects of the question remain unanswered?
- What new leads do the passages suggest?
- Are there connections between passages that need more investigation?

Respond with ONLY the JSON object. No other text."#;

#[derive(Debug)]
pub enum ReaderDecision {
    Continue { follow_up_queries: Vec<String> },
    Synthesize,
}

#[derive(serde::Deserialize)]
struct ReaderOutput {
    decision: String,
    follow_up_queries: Option<Vec<String>>,
}

pub struct Reader {
    llm: AnthropicClient,
    model: String,
}

impl Reader {
    pub fn new(llm: AnthropicClient, model: String) -> Self {
        Self { llm, model }
    }

    pub async fn read(
        &self,
        question: &str,
        new_passages: &[String],
        accumulated_context: &[String],
    ) -> Result<(ReaderDecision, LlmResponse)> {
        let passages_text = new_passages
            .iter()
            .enumerate()
            .map(|(i, p)| format!("[Passage {}] {}", i + 1, p))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Summarize accumulated context to keep prompt size manageable
        let context_summary = if accumulated_context.len() > 5 {
            format!(
                "{} passages accumulated so far. Latest 5:\n{}",
                accumulated_context.len(),
                accumulated_context
                    .iter()
                    .rev()
                    .take(5)
                    .rev()
                    .enumerate()
                    .map(|(i, p)| format!("[Context {}] {}", i + 1, truncate(p, 200)))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        } else {
            accumulated_context
                .iter()
                .enumerate()
                .map(|(i, p)| format!("[Context {}] {}", i + 1, truncate(p, 200)))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let user_message = format!(
            "Question: {}\n\nNew Passages:\n{}\n\nAccumulated Context:\n{}",
            question, passages_text, context_summary
        );

        let response = self
            .llm
            .complete(&self.model, Some(SYSTEM_PROMPT), &user_message)
            .await?;

        let decision = parse_decision(&response.text);

        Ok((decision, response))
    }
}

fn parse_decision(text: &str) -> ReaderDecision {
    // Try to parse the JSON response
    let json_str = if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            &text[start..=end]
        } else {
            text
        }
    } else {
        text
    };

    if let Ok(output) = serde_json::from_str::<ReaderOutput>(json_str) {
        if output.decision == "continue" {
            if let Some(queries) = output.follow_up_queries {
                if !queries.is_empty() {
                    return ReaderDecision::Continue {
                        follow_up_queries: queries,
                    };
                }
            }
        }
    }

    // Default to synthesize if parsing fails or decision is "synthesize"
    ReaderDecision::Synthesize
}

fn truncate(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        s
    } else {
        &s[..s.floor_char_boundary(max_chars)]
    }
}
