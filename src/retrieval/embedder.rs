use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let model_type = match model_name {
            "sentence-transformers/all-MiniLM-L6-v2" | "all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            "mixedbread-ai/mxbai-embed-large-v1" => EmbeddingModel::MxbaiEmbedLargeV1,
            "nomic-ai/nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
            _ => anyhow::bail!("Unsupported embedding model: {}", model_name),
        };

        let model = TextEmbedding::try_new(InitOptions::new(model_type).with_show_download_progress(true))
            .context("Failed to initialize embedding model")?;

        Ok(Self { model })
    }

    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let embeddings = self
            .model
            .embed(texts, None)
            .context("Failed to generate embeddings")?;
        Ok(embeddings)
    }
}
