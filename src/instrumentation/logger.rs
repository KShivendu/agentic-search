use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopLog {
    pub hop_number: u32,
    pub queries: Vec<String>,
    pub embedding_latency_ms: u64,
    pub search_latency_ms: u64,
    pub num_results: u32,
    pub tokens_in_passages: u32,
    pub llm_latency_ms: u64,
    pub llm_input_tokens: u32,
    pub llm_output_tokens: u32,
    pub llm_cost: f64,
    pub decision: String,
    pub total_hop_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLog {
    pub id: String,
    pub timestamp: String,
    pub question: String,
    pub hops: Vec<HopLog>,
    pub synthesis_latency_ms: u64,
    pub synthesis_input_tokens: u32,
    pub synthesis_output_tokens: u32,
    pub plan_latency_ms: u64,
    pub plan_input_tokens: u32,
    pub plan_output_tokens: u32,
    pub total_latency_ms: u64,
    pub total_llm_input_tokens: u32,
    pub total_llm_output_tokens: u32,
    pub total_cost: f64,
    pub final_answer: String,
}

impl RunLog {
    pub fn total_tokens(&self) -> u32 {
        self.total_llm_input_tokens + self.total_llm_output_tokens
    }

    /// Actual cost in USD as reported by the LLM API (e.g. OpenRouter usage.cost).
    pub fn cost(&self) -> f64 {
        self.total_cost
    }

    pub fn summary(&self) -> String {
        format!(
            "Hops: {} | Total latency: {:.1}s | Tokens retrieved: {} | Tokens used by LLM: {} | Cost: ${:.4}",
            self.hops.len(),
            self.total_latency_ms as f64 / 1000.0,
            self.hops.iter().map(|h| h.tokens_in_passages).sum::<u32>(),
            self.total_tokens(),
            self.cost(),
        )
    }
}

pub struct RunLogger {
    dir: PathBuf,
}

impl RunLogger {
    pub fn new(dir: &str) -> Result<Self> {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(&dir).context("Failed to create logs directory")?;
        Ok(Self { dir })
    }

    pub fn write(&self, run_log: &RunLog) -> Result<()> {
        let path = self.dir.join("runs.jsonl");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .context("Failed to open log file")?;

        let json = serde_json::to_string(run_log).context("Failed to serialize run log")?;
        writeln!(file, "{}", json).context("Failed to write log")?;

        Ok(())
    }
}
