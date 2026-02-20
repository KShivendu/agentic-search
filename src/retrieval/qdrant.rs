use anyhow::{Context, Result};
use qdrant_client::qdrant::{Document, Query, QueryPointsBuilder, ScoredPoint};
use qdrant_client::Qdrant;

#[derive(Debug, Clone)]
pub struct Passage {
    pub text: String,
}

pub struct QdrantRetriever {
    client: Qdrant,
    collection: String,
    embedding_model: String,
}

impl QdrantRetriever {
    pub async fn new(
        url: &str,
        api_key: Option<&str>,
        collection: &str,
        embedding_model: &str,
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
        })
    }

    /// Search using Qdrant cloud inference (server-side embedding).
    pub async fn search(&self, query_text: &str, top_k: u64) -> Result<Vec<Passage>> {
        let results = self
            .client
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
            .context("Qdrant query failed")?;

        Ok(Self::extract_passages(results.result))
    }

    fn extract_passages(points: Vec<ScoredPoint>) -> Vec<Passage> {
        points
            .into_iter()
            .map(|point| {
                let payload = point.payload;
                let text = payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                Passage { text }
            })
            .collect()
    }
}
