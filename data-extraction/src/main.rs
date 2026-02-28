mod extractors;
mod mixer;
mod types;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Training data extraction pipeline for Light Architects foundation model.
/// Extracts structured data from SOUL vault, transcripts, QUANTUM corpus,
/// KJV Bible, trace data, and MCP tool schemas.
#[derive(Parser)]
#[command(name = "data-extract")]
#[command(version = "0.1.0")]
#[command(about = "Training data extraction for Light Architects LLM")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract training data from one or more sources
    Extract {
        /// Data source to extract from
        #[arg(long, value_parser = parse_source)]
        source: Source,

        /// Output directory for extracted data
        #[arg(
            long,
            default_value = "/Users/kft/Projects/LightArchitectsFoundationModel/training-data"
        )]
        output_dir: PathBuf,
    },
    /// Mix all extracted datasets into train/val/test splits
    Mix {
        /// Directory containing extracted JSON files
        #[arg(
            long,
            default_value = "/Users/kft/Projects/LightArchitectsFoundationModel/training-data"
        )]
        output_dir: PathBuf,
    },
    /// Print statistics for a dataset file
    Stats {
        /// Path to the JSON dataset file
        #[arg(long)]
        input: String,
    },
}

#[derive(Clone, Debug)]
enum Source {
    Helix,
    Transcript,
    Quantum,
    Bible,
    Trace,
    Tools,
    All,
}

fn parse_source(s: &str) -> Result<Source, String> {
    match s.to_lowercase().as_str() {
        "helix" => Ok(Source::Helix),
        "transcript" => Ok(Source::Transcript),
        "quantum" => Ok(Source::Quantum),
        "bible" => Ok(Source::Bible),
        "trace" => Ok(Source::Trace),
        "tools" => Ok(Source::Tools),
        "all" => Ok(Source::All),
        _ => Err(format!(
            "Unknown source '{}'. Valid: helix, transcript, quantum, \
             bible, trace, tools, all",
            s
        )),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Extract { source, output_dir } => run_extract(&source, &output_dir),
        Commands::Mix { output_dir } => {
            mixer::mix(Some(
                output_dir.to_str().context("invalid output dir path")?,
            ))?;
            Ok(())
        }
        Commands::Stats { input } => mixer::print_file_stats(&input),
    }
}

fn run_extract(source: &Source, output_dir: &PathBuf) -> Result<()> {
    // Ensure output directory exists
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output directory {}", output_dir.display()))?;

    match source {
        Source::Helix => extract_helix(output_dir),
        Source::Transcript => extract_transcript(output_dir),
        Source::Quantum => extract_quantum(output_dir),
        Source::Bible => extract_bible(output_dir),
        Source::Trace => extract_trace(output_dir),
        Source::Tools => extract_tools(output_dir),
        Source::All => extract_all(output_dir),
    }
}

fn extract_helix(output_dir: &PathBuf) -> Result<()> {
    let samples = extractors::helix::extract(None)?;
    write_output(output_dir, "helix-alpaca.json", &samples)?;
    eprintln!("[extract] Wrote {} helix samples", samples.len());
    Ok(())
}

fn extract_transcript(output_dir: &PathBuf) -> Result<()> {
    let samples = extractors::transcript::extract(None)?;
    write_output(output_dir, "transcripts-sharegpt.json", &samples)?;
    eprintln!("[extract] Wrote {} transcript conversations", samples.len());
    Ok(())
}

fn extract_quantum(output_dir: &PathBuf) -> Result<()> {
    let samples = extractors::quantum::extract(None)?;
    write_output(output_dir, "quantum-alpaca.json", &samples)?;
    eprintln!("[extract] Wrote {} quantum samples", samples.len());
    Ok(())
}

fn extract_bible(output_dir: &PathBuf) -> Result<()> {
    let samples = extractors::bible::extract(None)?;
    write_output(output_dir, "bible-alpaca.json", &samples)?;
    eprintln!("[extract] Wrote {} bible samples", samples.len());
    Ok(())
}

fn extract_trace(output_dir: &PathBuf) -> Result<()> {
    let samples = extractors::trace::extract(None)?;
    write_output(output_dir, "traces-alpaca.json", &samples)?;
    eprintln!("[extract] Wrote {} trace samples", samples.len());
    Ok(())
}

fn extract_tools(output_dir: &PathBuf) -> Result<()> {
    let samples = extractors::tool_schema::extract()?;
    write_output(output_dir, "tool-schemas-alpaca.json", &samples)?;
    eprintln!("[extract] Wrote {} tool schema samples", samples.len());
    Ok(())
}

fn extract_all(output_dir: &PathBuf) -> Result<()> {
    eprintln!("[extract] Running all extractors...\n");

    extract_helix(output_dir)?;
    extract_transcript(output_dir)?;
    extract_quantum(output_dir)?;
    extract_bible(output_dir)?;
    extract_trace(output_dir)?;
    extract_tools(output_dir)?;

    eprintln!("\n[extract] All sources extracted.");
    Ok(())
}

fn write_output<T: serde::Serialize>(output_dir: &PathBuf, filename: &str, data: &T) -> Result<()> {
    let path = output_dir.join(filename);
    let json = serde_json::to_string_pretty(data).context("serializing to JSON")?;
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
