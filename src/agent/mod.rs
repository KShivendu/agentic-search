pub mod planner;
pub mod reader;
pub mod synthesizer;

use anyhow::Result;
use std::time::Instant;

use crate::config::Config;
use crate::instrumentation::{HopLog, RunLog, RunLogger};
use crate::llm::LlmClient;
use crate::retrieval::QdrantRetriever;

use planner::Planner;
use reader::{Reader, ReaderDecision};
use synthesizer::Synthesizer;

pub struct Agent {
    planner: Planner,
    reader: Reader,
    synthesizer: Synthesizer,
    retriever: QdrantRetriever,
    config: Config,
    logger: RunLogger,
}

impl Agent {
    pub async fn new(config: Config) -> Result<Self> {
        let llm = LlmClient::new(&config.llm_api_key, &config.llm_base_url);
        let retriever = QdrantRetriever::new(
            &config.qdrant_url,
            config.qdrant_api_key.as_deref(),
            &config.qdrant_collection,
            &config.embedding_model,
        )
        .await?;
        let logger = RunLogger::new("logs")?;

        Ok(Self {
            planner: Planner::new(llm.clone(), config.planner_model.clone()),
            reader: Reader::new(llm.clone(), config.reader_model.clone()),
            synthesizer: Synthesizer::new(llm, config.synthesizer_model.clone()),
            retriever,
            config,
            logger,
        })
    }

    pub async fn ask(&self, question: &str, verbose: bool) -> Result<RunLog> {
        let run_start = Instant::now();
        let mut hops: Vec<HopLog> = Vec::new();
        let mut accumulated_context: Vec<String> = Vec::new();

        // Step 1: Plan initial queries
        let plan_start = Instant::now();
        let (queries, plan_response) = self.planner.plan(question).await?;
        let plan_latency = plan_start.elapsed().as_millis() as u64;

        if verbose {
            eprintln!(
                "[planner] Generated {} queries in {}ms",
                queries.len(),
                plan_latency
            );
            for q in &queries {
                eprintln!("  - {}", q);
            }
        }

        let mut pending_queries = queries;

        for hop_number in 0..self.config.max_hops {
            if pending_queries.is_empty() {
                break;
            }

            let hop_start = Instant::now();

            // Search Qdrant (cloud inference handles embedding server-side)
            let search_start = Instant::now();
            let query_text = pending_queries.join(" ");
            let passages = self
                .retriever
                .search(&query_text, self.config.top_k)
                .await?;
            let search_latency = search_start.elapsed().as_millis() as u64;
            let num_results = passages.len();

            let passage_texts: Vec<String> = passages.iter().map(|p| p.text.clone()).collect();
            let tokens_in_passages: u32 = passage_texts.iter().map(|t| (t.len() / 4) as u32).sum();

            accumulated_context.extend(passage_texts.clone());

            // Reader decides: continue or synthesize
            let llm_start = Instant::now();
            let (decision, reader_response) = self
                .reader
                .read(question, &passage_texts, &accumulated_context)
                .await?;
            let llm_latency = llm_start.elapsed().as_millis() as u64;

            let hop_log = HopLog {
                hop_number: hop_number as u32,
                queries: pending_queries.clone(),
                embedding_latency_ms: 0,
                search_latency_ms: search_latency,
                num_results: num_results as u32,
                tokens_in_passages,
                llm_latency_ms: llm_latency,
                llm_input_tokens: reader_response.input_tokens,
                llm_output_tokens: reader_response.output_tokens,
                llm_cost: reader_response.cost,
                decision: match &decision {
                    ReaderDecision::Continue { follow_up_queries } => {
                        format!("continue({})", follow_up_queries.len())
                    }
                    ReaderDecision::Synthesize => "synthesize".into(),
                },
                total_hop_latency_ms: hop_start.elapsed().as_millis() as u64,
            };

            if verbose {
                eprintln!(
                    "[hop {}] {} results, search={}ms llm={}ms → {}",
                    hop_number, num_results, search_latency, llm_latency, hop_log.decision
                );
            }

            hops.push(hop_log);

            match decision {
                ReaderDecision::Continue { follow_up_queries } => {
                    pending_queries = follow_up_queries;
                }
                ReaderDecision::Synthesize => break,
            }
        }

        // Synthesize final answer
        let synth_start = Instant::now();
        let (answer, synth_response) = self
            .synthesizer
            .synthesize(question, &accumulated_context)
            .await?;
        let synth_latency = synth_start.elapsed().as_millis() as u64;

        if verbose {
            eprintln!("[synthesizer] Generated answer in {}ms", synth_latency);
        }

        let total_latency = run_start.elapsed().as_millis() as u64;
        let total_llm_input_tokens: u32 = plan_response.input_tokens
            + synth_response.input_tokens
            + hops.iter().map(|h| h.llm_input_tokens).sum::<u32>();
        let total_llm_output_tokens: u32 = plan_response.output_tokens
            + synth_response.output_tokens
            + hops.iter().map(|h| h.llm_output_tokens).sum::<u32>();
        let total_cost: f64 =
            plan_response.cost + synth_response.cost + hops.iter().map(|h| h.llm_cost).sum::<f64>();

        let run_log = RunLog {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            question: question.to_string(),
            hops: hops.clone(),
            synthesis_latency_ms: synth_latency,
            synthesis_input_tokens: synth_response.input_tokens,
            synthesis_output_tokens: synth_response.output_tokens,
            plan_latency_ms: plan_latency,
            plan_input_tokens: plan_response.input_tokens,
            plan_output_tokens: plan_response.output_tokens,
            total_latency_ms: total_latency,
            total_llm_input_tokens,
            total_llm_output_tokens,
            total_cost,
            final_answer: answer,
        };

        self.logger.write(&run_log)?;

        Ok(run_log)
    }
}
