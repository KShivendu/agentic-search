use anyhow::Result;

use crate::llm::{LlmClient, LlmResponse};

const SYSTEM_PROMPT: &str = r#"You are a research reader. You are given a question, retrieved passages, and context accumulated from previous research hops.

Your job is to decide whether you have enough information to answer the question.

Always respond with a JSON object with these fields:
- "reasoning": 1-2 sentences — what relevant facts you found and what specific gap remains (or why you have enough)
- "decision": "synthesize" or "continue"
- "follow_up_queries": (only when decision is "continue") 1-3 queries targeting the specific gap named in reasoning

Examples:
{"reasoning": "Found that transistors replaced vacuum tubes in spacecraft by 1960, but nothing on NASA procurement contracts or cost figures.", "decision": "continue", "follow_up_queries": ["NASA transistor procurement budget 1960s", "cost comparison vacuum tubes vs transistors spacecraft"]}
{"reasoning": "Passages cover both the timeline and the key programs (Apollo, Ranger) with sufficient technical detail to answer the question.", "decision": "synthesize"}

Respond with ONLY the JSON object. No other text."#;

#[derive(Debug)]
pub enum ReaderDecision {
    Continue { follow_up_queries: Vec<String> },
    Synthesize,
}

#[derive(serde::Deserialize)]
struct ReaderOutput {
    reasoning: Option<String>,
    decision: String,
    follow_up_queries: Option<Vec<String>>,
}

pub struct Reader {
    llm: LlmClient,
    model: String,
}

impl Reader {
    pub fn new(llm: LlmClient, model: String) -> Self {
        Self { llm, model }
    }

    pub async fn read(
        &self,
        question: &str,
        new_passages: &[String],
        accumulated_context: &[String],
    ) -> Result<(ReaderDecision, Option<String>, LlmResponse)> {
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

        let (decision, reasoning) = parse_decision(&response.text);

        Ok((decision, reasoning, response))
    }
}

fn parse_decision(text: &str) -> (ReaderDecision, Option<String>) {
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
        let reasoning = output.reasoning;
        if output.decision == "continue" {
            if let Some(queries) = output.follow_up_queries {
                if !queries.is_empty() {
                    return (
                        ReaderDecision::Continue {
                            follow_up_queries: queries,
                        },
                        reasoning,
                    );
                }
            }
        }
        return (ReaderDecision::Synthesize, reasoning);
    }

    (ReaderDecision::Synthesize, None)
}

fn truncate(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        s
    } else {
        &s[..s.floor_char_boundary(max_chars)]
    }
}
