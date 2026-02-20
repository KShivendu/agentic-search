mod agent;
mod config;
mod instrumentation;
mod llm;
mod retrieval;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::BufRead;

use agent::Agent;
use config::Config;
use instrumentation::RunLog;

#[derive(Parser)]
#[command(name = "agentic-search", about = "Multi-hop research agent over a large corpus")]
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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let config = Config::from_env()?;
    let agent = Agent::new(config).await?;

    match cli.command {
        Commands::Ask { question } => {
            let run_log = agent.ask(&question, cli.verbose).await?;
            println!("\n{}\n", run_log.final_answer);
            println!("{}", run_log.summary());
        }
        Commands::Eval { path } => {
            let file =
                std::fs::File::open(&path).context(format!("Failed to open eval file: {}", path))?;
            let reader = std::io::BufReader::new(file);

            let mut run_logs: Vec<RunLog> = Vec::new();
            let mut errors = 0;

            for (i, line) in reader.lines().enumerate() {
                let line = line.context("Failed to read line")?;
                if line.trim().is_empty() {
                    continue;
                }

                let eq: EvalQuestion =
                    serde_json::from_str(&line).context(format!("Failed to parse line {}", i + 1))?;

                eprintln!("\n[{}/...] {}", i + 1, eq.question);

                match agent.ask(&eq.question, cli.verbose).await {
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

                let avg_hops =
                    run_logs.iter().map(|r| r.hops.len()).sum::<usize>() as f64 / run_logs.len() as f64;
                let avg_latency = run_logs.iter().map(|r| r.total_latency_ms).sum::<u64>() as f64
                    / run_logs.len() as f64;
                let total_tokens: u32 = run_logs.iter().map(|r| r.total_tokens()).sum();
                let total_cost: f64 = run_logs.iter().map(|r| r.estimated_cost()).sum();

                println!("Avg hops: {:.1}", avg_hops);
                println!("Avg latency: {:.1}s", avg_latency / 1000.0);
                println!("Total tokens: {}", total_tokens);
                println!("Total cost: ${:.4}", total_cost);
            }
        }
    }

    Ok(())
}
