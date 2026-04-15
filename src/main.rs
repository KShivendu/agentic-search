mod agent;
mod config;
mod instrumentation;
mod llm;
mod retrieval;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::io::BufRead;

use agent::Agent;
use config::Config;
use instrumentation::RunLog;

#[derive(Parser)]
#[command(
    name = "agentic-search",
    about = "Multi-hop research agent over a large corpus"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose per-hop output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Ask a single question
    Ask {
        /// The question to research
        question: String,
    },
    /// Run evaluation on a question set
    Eval {
        /// Path to JSONL file with questions
        path: String,
    },
}

#[derive(serde::Deserialize)]
struct EvalQuestion {
    question: String,
    #[allow(dead_code)]
    expected_answer: Option<String>,
    #[allow(dead_code)]
    r#type: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let config = Config::from_env()?;
    let agent = Agent::new(config).await?;

    match cli.command {
        Commands::Ask { question } => {
            let run_log = agent.ask(&question, cli.verbose, true).await?;
            // Print source legend: highlight only sources cited in the answer
            if !run_log.sources.is_empty() {
                let cited: std::collections::HashSet<usize> = cited_indices(&run_log.final_answer);
                let any_cited = run_log
                    .sources
                    .iter()
                    .enumerate()
                    .any(|(i, src)| !src.title.is_empty() && cited.contains(&(i + 1)));
                let header = if any_cited {
                    "Sources:"
                } else {
                    "Sources (none cited):"
                };
                eprintln!("\n{}", header.dimmed());
                for (i, src) in run_log.sources.iter().enumerate() {
                    if src.title.is_empty() {
                        continue;
                    }
                    let n = i + 1;
                    let chunk_str = match src.chunk_index {
                        Some(idx) => format!(" (chunk {})", idx),
                        None => String::new(),
                    };
                    if cited.contains(&n) {
                        eprintln!(
                            "  {} {}{}",
                            format!("[{}]", n).cyan().bold(),
                            src.title,
                            chunk_str.dimmed(),
                        );
                    } else if cli.verbose {
                        eprintln!(
                            "  {} {}{}",
                            format!("[{}]", n).dimmed(),
                            src.title.dimmed(),
                            chunk_str.dimmed(),
                        );
                    }
                }
            }
            eprintln!("\n{}", run_log.summary().dimmed());
        }
        Commands::Eval { path } => {
            let file = std::fs::File::open(&path)
                .context(format!("Failed to open eval file: {}", path))?;
            let reader = std::io::BufReader::new(file);

            let mut run_logs: Vec<RunLog> = Vec::new();
            let mut errors = 0;

            for (i, line) in reader.lines().enumerate() {
                let line = line.context("Failed to read line")?;
                if line.trim().is_empty() {
                    continue;
                }

                let eq: EvalQuestion = serde_json::from_str(&line)
                    .context(format!("Failed to parse line {}", i + 1))?;

                eprintln!("\n[{}/...] {}", i + 1, eq.question);

                match agent.ask(&eq.question, cli.verbose, false).await {
                    Ok(run_log) => {
                        println!("  {}", run_log.summary());
                        run_logs.push(run_log);
                    }
                    Err(e) => {
                        eprintln!("  ERROR: {}", e);
                        errors += 1;
                    }
                }
            }

            if !run_logs.is_empty() {
                println!("\n=== Evaluation Summary ===");
                println!("Questions: {} (errors: {})", run_logs.len(), errors);

                let avg_hops = run_logs.iter().map(|r| r.hops.len()).sum::<usize>() as f64
                    / run_logs.len() as f64;
                let avg_latency = run_logs.iter().map(|r| r.total_latency_ms).sum::<u64>() as f64
                    / run_logs.len() as f64;
                let total_tokens: u32 = run_logs.iter().map(|r| r.total_tokens()).sum();
                let total_cost: f64 = run_logs.iter().map(|r| r.cost()).sum();

                println!("Avg hops: {:.1}", avg_hops);
                println!("Avg latency: {:.1}s", avg_latency / 1000.0);
                println!("Total tokens: {}", total_tokens);
                println!("Total cost: ${:.4}", total_cost);
            }
        }
    }

    Ok(())
}

/// Parse all `[N]` citation numbers from the answer text (1-based).
fn cited_indices(text: &str) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && j < bytes.len() && bytes[j] == b']' {
                if let Ok(n) = text[start..j].parse::<usize>() {
                    set.insert(n);
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    set
}
