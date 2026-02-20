use anyhow::{Context, Result};
use qdrant_client::qdrant::SearchPoints;
use qdrant_client::Qdrant;

#[derive(Debug, Clone)]
pub struct Passage {
    pub text: String,
    pub title: String,
    pub score: f32,
}

pub struct QdrantRetriever {
    client: Qdrant,
    collection: String,
}

impl QdrantRetriever {
    pub async fn new(url: &str, collection: &str) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .context("Failed to connect to Qdrant")?;

        Ok(Self {
            client,
            collection: collection.to_string(),
        })
    }

    pub async fn search(&self, query_vector: &[f32], top_k: u64) -> Result<Vec<Passage>> {
        let search_request = SearchPoints {
            collection_name: self.collection.clone(),
            vector: query_vector.to_vec(),
            limit: top_k,
            with_payload: Some(true.into()),
            ..Default::default()
        };

        let results = self
            .client
            .search_points(search_request)
            .await
            .context("Qdrant search failed")?;

        let passages = results
            .result
            .into_iter()
            .map(|point| {
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

                Passage {
                    text,
                    title,
                    score: point.score,
                }
            })
            .collect();

        Ok(passages)
    }
}
