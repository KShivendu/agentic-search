use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub llm_api_key: String,
    pub llm_base_url: String,
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub qdrant_collection: String,
    pub planner_model: String,
    pub reader_model: String,
    pub synthesizer_model: String,
    pub embedding_model: String,
    pub cloud_inference: bool,
    pub max_hops: usize,
    pub top_k: u64,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Self {
            llm_api_key: std::env::var("LLM_API_KEY")
                .context("LLM_API_KEY must be set")?,
            llm_base_url: std::env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".into()),
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6334".into()),
            qdrant_api_key: std::env::var("QDRANT_API_KEY").ok(),
            qdrant_collection: std::env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "wiki_passages".into()),
            planner_model: std::env::var("PLANNER_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-haiku-4-5-20241022".into()),
            reader_model: std::env::var("READER_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-haiku-4-5-20241022".into()),
            synthesizer_model: std::env::var("SYNTHESIZER_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-sonnet-4-20250514".into()),
            embedding_model: std::env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "mixedbread-ai/mxbai-embed-large-v1".into()),
            cloud_inference: std::env::var("CLOUD_INFERENCE")
                .unwrap_or_else(|_| "true".into())
                .parse()
                .unwrap_or(true),
            max_hops: std::env::var("MAX_HOPS")
                .unwrap_or_else(|_| "7".into())
                .parse()
                .context("MAX_HOPS must be a number")?,
            top_k: std::env::var("TOP_K")
                .unwrap_or_else(|_| "10".into())
                .parse()
                .context("TOP_K must be a number")?,
        })
    }
}
