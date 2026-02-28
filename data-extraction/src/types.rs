use serde::{Deserialize, Serialize};

/// Alpaca-format training sample (instruction/input/output triple)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlpacaSample {
    pub instruction: String,
    pub input: String,
    pub output: String,
    pub source: String,
}

/// ShareGPT-format training sample (multi-turn conversation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareGptSample {
    pub conversations: Vec<Turn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A single turn in a ShareGPT conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub from: String,
    pub value: String,
}

/// YAML frontmatter parsed from a helix entry.
/// All fields are needed for deserialization even if not all
/// are read directly by every extractor.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct HelixFrontmatter {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub sibling: Option<String>,
    #[serde(default)]
    pub significance: Option<f64>,
    #[serde(default)]
    pub strands: Vec<String>,
    #[serde(default)]
    pub emotions: Vec<String>,
    #[serde(default)]
    pub themes: Vec<String>,
    #[serde(default)]
    pub epoch: Option<String>,
    #[serde(default)]
    pub self_defining: Option<bool>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Statistics for a dataset
#[derive(Debug, Default, Serialize)]
pub struct DatasetStats {
    pub total_samples: usize,
    pub source_counts: std::collections::HashMap<String, usize>,
    pub avg_output_chars: f64,
    pub min_output_chars: usize,
    pub max_output_chars: usize,
}

/// Trace decision point from trace-engine JSON.
/// All fields are needed for deserialization even if not all
/// are read directly.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct TraceEntry {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub sibling: Option<String>,
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub decision_points: Vec<DecisionPoint>,
    #[serde(default)]
    pub strand_activations: Vec<StrandActivation>,
    #[serde(default)]
    pub outcome: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionPoint {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub decision: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StrandActivation {
    #[serde(default)]
    pub strand: Option<String>,
    #[serde(default)]
    pub weight: Option<f64>,
}
