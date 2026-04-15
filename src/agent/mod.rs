pub mod planner;
pub mod reader;
pub mod synthesizer;

use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use std::time::Instant;

use futures_util::future::join_all;

use crate::config::Config;
use crate::instrumentation::{HopLog, RunLog, RunLogger, SourceRef};
use crate::llm::{LlmClient, StreamEvent};
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

fn new_spinner(msg: &str) -> ProgressBar {
    let sp = ProgressBar::new_spinner();
    sp.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    sp.set_message(msg.to_string());
    sp.enable_steady_tick(std::time::Duration::from_millis(80));
    sp
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

    pub async fn ask(&self, question: &str, verbose: bool, stream: bool) -> Result<RunLog> {
        let run_start = Instant::now();
        let mut hops: Vec<HopLog> = Vec::new();
        let mut accumulated_context: Vec<String> = Vec::new();
        let mut source_refs: Vec<SourceRef> = Vec::new();

        // Step 1: Plan initial queries
        let spinner = if stream {
            Some(new_spinner("Planning queries..."))
        } else {
            None
        };

        let plan_start = Instant::now();
        let (queries, plan_response) = self.planner.plan(question).await?;
        let plan_latency = plan_start.elapsed().as_millis() as u64;

        if let Some(sp) = &spinner {
            sp.finish_and_clear();
        }

        if stream {
            eprintln!(
                "{} {} queries {}",
                "Planning".cyan().bold(),
                queries.len(),
                format!("({}ms)", plan_latency).dimmed(),
            );
            for (i, q) in queries.iter().enumerate() {
                eprintln!("  {}  {}", format!("{}.", i + 1).dimmed(), q);
            }
            eprintln!();
        }

        let mut pending_queries = queries;

        for hop_number in 0..self.config.max_hops {
            if pending_queries.is_empty() {
                break;
            }

            let hop_start = Instant::now();

            // Search Qdrant
            let spinner = if stream {
                Some(new_spinner(&format!(
                    "Searching (hop {})...",
                    hop_number + 1
                )))
            } else {
                None
            };

            let search_start = Instant::now();
            let search_futures = pending_queries.iter().map(|q| {
                let retriever = &self.retriever;
                let top_k = self.config.top_k;
                async move {
                    let t = Instant::now();
                    let result = retriever.search(q, top_k).await;
                    (result, t.elapsed().as_millis() as u64)
                }
            });
            let search_results = join_all(search_futures).await;
            let search_latency = search_start.elapsed().as_millis() as u64;

            // Merge results from all queries, deduplicating by point_id
            let mut seen_ids = std::collections::HashSet::new();
            let mut passages = Vec::new();
            let mut per_query_latencies: Vec<u64> = Vec::new();
            for (result, latency) in search_results {
                per_query_latencies.push(latency);
                for passage in result? {
                    if seen_ids.insert(passage.point_id.clone()) {
                        passages.push(passage);
                    }
                }
            }
            let num_results = passages.len();

            if let Some(sp) = &spinner {
                sp.set_message(format!("Reading (hop {})...", hop_number + 1));
            }

            let passage_texts: Vec<String> = passages.iter().map(|p| p.text.clone()).collect();
            let tokens_in_passages: u32 = passage_texts.iter().map(|t| (t.len() / 4) as u32).sum();

            accumulated_context.extend(passage_texts.clone());
            source_refs.extend(passages.iter().map(|p| SourceRef {
                point_id: p.point_id.clone(),
                title: p.title.clone(),
                chunk_index: p.chunk_index,
            }));

            // Reader decides: continue or synthesize
            let llm_start = Instant::now();
            let (decision, reasoning, reader_response) = self
                .reader
                .read(question, &passage_texts, &accumulated_context)
                .await?;
            let llm_latency = llm_start.elapsed().as_millis() as u64;

            if let Some(sp) = &spinner {
                sp.finish_and_clear();
            }

            let decision_str = match &decision {
                ReaderDecision::Continue { follow_up_queries } => {
                    format!("continue({})", follow_up_queries.len())
                }
                ReaderDecision::Synthesize => "synthesize".into(),
            };

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
                decision: decision_str.clone(),
                total_hop_latency_ms: hop_start.elapsed().as_millis() as u64,
            };

            if stream {
                let search_detail = per_query_latencies
                    .iter()
                    .map(|ms| format!("{}ms", ms))
                    .collect::<Vec<_>>()
                    .join(", ");
                let latency_info = if verbose {
                    format!(
                        " {}",
                        format!("(search [{}] read {}ms)", search_detail, llm_latency).dimmed()
                    )
                } else {
                    format!(" {}", format!("(search [{}])", search_detail).dimmed())
                };
                match &decision {
                    ReaderDecision::Continue { follow_up_queries } => {
                        eprintln!(
                            "{} {} passages → {} follow-up queries{}",
                            format!("Hop {}", hop_number + 1).cyan().bold(),
                            num_results,
                            follow_up_queries.len(),
                            latency_info,
                        );
                        if let Some(r) = &reasoning {
                            eprintln!("  {}", r.dimmed());
                        }
                        for (i, q) in follow_up_queries.iter().enumerate() {
                            eprintln!("  {}  {}", format!("{}.", i + 1).dimmed(), q);
                        }
                        eprintln!();
                    }
                    ReaderDecision::Synthesize => {
                        eprintln!(
                            "{} {} passages → ready to answer{}",
                            format!("Hop {}", hop_number + 1).cyan().bold(),
                            num_results,
                            latency_info,
                        );
                        if let Some(r) = &reasoning {
                            eprintln!("  {}", r.dimmed());
                        }
                        eprintln!();
                    }
                }
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

        let (answer, synth_input_tokens, synth_output_tokens, synth_cost) = if stream {
            let spinner = new_spinner("Synthesizing...");

            let mut rx = self
                .synthesizer
                .synthesize_stream(question, &accumulated_context)
                .await?;

            let mut full_text = String::new();
            let mut input_tokens = 0u32;
            let mut output_tokens = 0u32;
            let mut cost = 0.0f64;
            let mut first_token = true;

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(token) => {
                        if first_token {
                            spinner.finish_and_clear();
                            // Print a leading newline before the streamed answer
                            eprintln!();
                            first_token = false;
                        }
                        print!("{}", token);
                        std::io::stdout().flush().ok();
                    }
                    StreamEvent::Done {
                        full_text: text,
                        input_tokens: it,
                        output_tokens: ot,
                        cost: c,
                    } => {
                        if first_token {
                            spinner.finish_and_clear();
                        }
                        full_text = text;
                        input_tokens = it;
                        output_tokens = ot;
                        cost = c;
                    }
                }
            }

            // Trailing newline after streamed answer
            println!();

            (full_text, input_tokens, output_tokens, cost)
        } else {
            let (answer, synth_response) = self
                .synthesizer
                .synthesize(question, &accumulated_context)
                .await?;
            (
                answer,
                synth_response.input_tokens,
                synth_response.output_tokens,
                synth_response.cost,
            )
        };

        let synth_latency = synth_start.elapsed().as_millis() as u64;

        if verbose {
            eprintln!(
                "{} Generated answer in {}",
                "[synthesizer]".cyan().bold(),
                format!("{}ms", synth_latency).yellow(),
            );
        }

        let total_latency = run_start.elapsed().as_millis() as u64;
        let total_llm_input_tokens: u32 = plan_response.input_tokens
            + synth_input_tokens
            + hops.iter().map(|h| h.llm_input_tokens).sum::<u32>();
        let total_llm_output_tokens: u32 = plan_response.output_tokens
            + synth_output_tokens
            + hops.iter().map(|h| h.llm_output_tokens).sum::<u32>();
        let total_cost: f64 =
            plan_response.cost + synth_cost + hops.iter().map(|h| h.llm_cost).sum::<f64>();

        let run_log = RunLog {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            question: question.to_string(),
            hops: hops.clone(),
            synthesis_latency_ms: synth_latency,
            synthesis_input_tokens: synth_input_tokens,
            synthesis_output_tokens: synth_output_tokens,
            plan_latency_ms: plan_latency,
            plan_input_tokens: plan_response.input_tokens,
            plan_output_tokens: plan_response.output_tokens,
            total_latency_ms: total_latency,
            total_llm_input_tokens,
            total_llm_output_tokens,
            total_cost,
            final_answer: answer,
            sources: source_refs,
            passages: accumulated_context,
        };

        self.logger.write(&run_log)?;

        Ok(run_log)
    }
}
