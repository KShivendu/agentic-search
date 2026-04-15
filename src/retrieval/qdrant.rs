use anyhow::{Context, Result};
use qdrant_client::qdrant::{Document, Query, QueryPointsBuilder, ScoredPoint};
use qdrant_client::Qdrant;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Passage {
    pub point_id: String,
    pub text: String,
    pub title: String,
    pub chunk_index: Option<u64>,
}

pub struct QdrantRetriever {
    client: Qdrant,
    collection: String,
    embedding_model: String,
    cloud_inference: bool,
    http_client: reqwest::Client,
    embedding_api_key: String,
    embedding_base_url: String,
}

// OpenAI-compatible response (OpenRouter, etc.)
#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingData {
    embedding: Vec<f32>,
}

// Cohere v2 response
#[derive(Deserialize)]
struct CohereEmbeddingResponse {
    embeddings: CohereEmbeddings,
}

#[derive(Deserialize)]
struct CohereEmbeddings {
    float: Vec<Vec<f32>>,
}

impl QdrantRetriever {
    pub async fn new(
        url: &str,
        api_key: Option<&str>,
        collection: &str,
        embedding_model: &str,
        cloud_inference: bool,
        embedding_api_key: &str,
        embedding_base_url: &str,
    ) -> Result<Self> {
        let mut builder = Qdrant::from_url(url);
        if let Some(key) = api_key {
            builder = builder.api_key(key);
        }
        let client = builder.build().context("Failed to connect to Qdrant")?;

        Ok(Self {
            client,
            collection: collection.to_string(),
            embedding_model: embedding_model.to_string(),
            cloud_inference,
            http_client: reqwest::Client::new(),
            embedding_api_key: embedding_api_key.to_string(),
            embedding_base_url: embedding_base_url.to_string(),
        })
    }

    async fn embed_via_http(&self, text: &str) -> Result<Vec<f32>> {
        if self.embedding_base_url.contains("cohere.com") {
            let body = serde_json::json!({
                "model": self.embedding_model,
                "texts": [text],
                "input_type": "search_query",
                "embedding_types": ["float"],
            });

            let response = self
                .http_client
                .post(&self.embedding_base_url)
                .bearer_auth(&self.embedding_api_key)
                .json(&body)
                .send()
                .await
                .context("Cohere embedding request failed")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("Cohere embedding API error {}: {}", status, body);
            }

            let response = response
                .json::<CohereEmbeddingResponse>()
                .await
                .context("Failed to parse Cohere embedding response")?;

            response
                .embeddings
                .float
                .into_iter()
                .next()
                .context("Cohere embedding API returned empty data")
        } else {
            // OpenAI-compatible (OpenRouter, etc.)
            let body = serde_json::json!({
                "model": self.embedding_model,
                "input": text,
            });

            let response = self
                .http_client
                .post(&self.embedding_base_url)
                .bearer_auth(&self.embedding_api_key)
                .json(&body)
                .send()
                .await
                .context("Embedding HTTP request failed")?
                .error_for_status()
                .context("Embedding API returned error status")?
                .json::<OpenAiEmbeddingResponse>()
                .await
                .context("Failed to parse embedding response")?;

            response
                .data
                .into_iter()
                .next()
                .map(|d| d.embedding)
                .context("Embedding API returned empty data")
        }
    }

    /// Search using cloud inference (server-side embedding) or OpenRouter HTTP embedding.
    pub async fn search(&self, query_text: &str, top_k: u64) -> Result<Vec<Passage>> {
        let results = if self.cloud_inference {
            self.client
                .query(
                    QueryPointsBuilder::new(&self.collection)
                        .query(Query::new_nearest(Document::new(
                            query_text,
                            &self.embedding_model,
                        )))
                        .limit(top_k)
                        .with_payload(true),
                )
                .await
                .context("Qdrant query failed")?
        } else {
            let vector = self.embed_via_http(query_text).await?;
            self.client
                .query(
                    QueryPointsBuilder::new(&self.collection)
                        .query(Query::new_nearest(vector))
                        .limit(top_k)
                        .with_payload(true),
                )
                .await
                .context("Qdrant query failed")?
        };

        Ok(Self::extract_passages(results.result))
    }

    fn extract_passages(points: Vec<ScoredPoint>) -> Vec<Passage> {
        use qdrant_client::qdrant::point_id::PointIdOptions;

        points
            .into_iter()
            .map(|point| {
                let point_id = point
                    .id
                    .and_then(|pid| pid.point_id_options)
                    .map(|opts| match opts {
                        PointIdOptions::Num(n) => n.to_string(),
                        PointIdOptions::Uuid(s) => s,
                    })
                    .unwrap_or_default();

                let payload = point.payload;
                let text = payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let title = payload
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let chunk_index = payload
                    .get("chunk_index")
                    .and_then(|v| v.as_integer())
                    .map(|i| i as u64);

                Passage {
                    point_id,
                    text,
                    title,
                    chunk_index,
                }
            })
            .collect()
    }
}
