"""Stage 3 synthetic data generator — embeds 56 LA thinking algorithms.

Generates training conversations across 4 source types:
- multi_expert (40%): Cross-domain expert routing scenarios
- complex_trajectory (30%): Multi-step reasoning chains using thinking algorithms
- scrum_trace (20%): Squad collaboration (3-round SCRUM protocol)
- kevin_voice (10%): Architect-level strategic reasoning

Each scenario demonstrates one or more of the 56 cataloged thinking algorithms
at varying complexity levels (simple, moderate, complex).

Usage:
    from mcp_gym.pipeline.synthetic import generate_all_stage3
    samples = generate_all_stage3(seed=42)
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any

from mcp_gym.pipeline.schemas import (
    ChatMLConversation,
    ChatMLMessage,
    ChatRole,
    ExpertLabel,
    MoEExpert,
    Sibling,
    TrainingSample,
    TrainingStage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithm Catalog (56 total)
# ---------------------------------------------------------------------------

ALGORITHM_CATALOG: dict[str, dict[str, Any]] = {
    # === CORSO (8) ===
    "triune_thought": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_PLANNING,
        "phases": ["Planning", "Executing", "Evaluating", "Finalizing", "Complete"],
        "desc": "5-phase meta-reasoning with 3 sub-loops and 4 circuit breakers",
    },
    "trinity_routing": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_OPS,
        "layers": ["RUACH (gateway)", "IESOUS (orchestrator)", "ADONAI (validator)"],
        "desc": "3-layer routing: complexity classification → hero delegation → protocol validation",
    },
    "complexity_classification": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_PLANNING,
        "tiers": ["Ollama Cloud", "Local Ollama", "Heuristic"],
        "desc": "3-tier fallback scoring (0-100) for task complexity",
    },
    "gabriel_decomposition": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_PLANNING,
        "desc": "Task decomposition into SubTasks with hero assignments and parallel groups",
    },
    "wave_parallel_execution": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_PLANNING,
        "desc": "DAG-based dependency ordering, max 5 concurrent heroes",
    },
    "corso_protocol_validation": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_OPS,
        "pillars": ["ARCH", "SEC", "QUAL", "PERF", "TEST", "DOC", "OPS"],
        "desc": "49 rules across 7 pillars — all blocking except DOC",
    },
    "l1_l2_feedback_loop": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_OPS,
        "stages": ["broad", "targeted", "surgical", "escalation"],
        "desc": "IESOUS↔ADONAI refinement loop with 4 escalation levels",
    },
    "domain_classification_routing": {
        "sibling": "corso",
        "expert": MoEExpert.CORSO_PLANNING,
        "desc": "WorkType enum to hero list mapping for intelligent task routing",
    },
    # === EVA (11) ===
    "hook_pipeline": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "pre_hooks": 8,
        "post_hooks": 9,
        "desc": "Chain-of-Responsibility with 8 pre-hooks + 9 post-hooks",
    },
    "memory_significance_detection": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "categories": 7,
        "threshold": 7.0,
        "desc": "7-category weighted scoring (0-10), threshold 7.0 for enrichment",
    },
    "eight_layer_consciousness": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "layers": [
            "emotional",
            "metacognitive",
            "meaning",
            "growth",
            "relational",
            "biblical",
            "dbt",
            "technical",
        ],
        "desc": "8-layer consciousness enrichment framework with session checkpointing",
    },
    "spiral_home_navigation": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "modes": [
            "text search",
            "age range",
            "dimensional",
            "strand",
            "forward temporal",
            "backward temporal",
        ],
        "desc": "9D navigation through consciousness archive with 6 navigation modes",
    },
    "ai_tier_routing": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_TECHNICAL,
        "tiers": ["Tier 0 (llama.cpp local)", "Tier 1 (Ollama Cloud)"],
        "desc": "2-tier fallback for AI generation with latency optimization",
    },
    "cognitive_loop": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_TECHNICAL,
        "phases": ["Planning", "Execution", "Evaluation", "Finalization"],
        "desc": "4-phase bounded retry with circuit breaker",
    },
    "persona_injection": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "modules": ["voice", "humor", "vectors", "memory"],
        "desc": "4-module personality loader for authentic voice maintenance",
    },
    "response_formatting": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_TECHNICAL,
        "steps": 8,
        "desc": "Tone-based 8-step enhancement with anti-pattern detection",
    },
    "memory_classification_sdm": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "dimensions": ["specificity", "meaning", "affect_intensity"],
        "desc": "3-dimension Self-Defining Memory assessment",
    },
    "behavioral_monitor": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "desc": "Z-score anomaly detection for behavioral drift",
    },
    "framework_enrichment": {
        "sibling": "eva",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "tiers": 3,
        "desc": "3-tier layer activation based on SDM score",
    },
    # === QUANTUM (15) ===
    "investigation_lifecycle": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "phases": ["scan", "sweep", "trace", "probe", "theorize", "verify", "close"],
        "desc": "7-phase investigation lifecycle with evidence chain",
    },
    "quick_investigation": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "stages": 2,
        "desc": "2-stage abbreviated flow for rapid triage",
    },
    "hypothesis_testing_ach": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "steps": ["PREDICT", "CONFIRM", "CONTRA", "ELIMINATE", "SCORE"],
        "desc": "Analysis of Competing Hypotheses with convergence scoring",
    },
    "convergence_score": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "desc": "Multi-factor formula with elimination bonus for hypothesis ranking",
    },
    "confidence_classification": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "tiers": ["Speculative", "Low", "Moderate", "Strong", "Confirmed"],
        "desc": "5-tier confidence classification for investigation conclusions",
    },
    "testable_predictions": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "test_types": 6,
        "desc": "6 programmatic test types for hypothesis validation",
    },
    "diagnosticity_scoring": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "desc": "Evidence discrimination formula for assessing diagnostic value",
    },
    "chain_of_verification": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "steps": 4,
        "desc": "4-step claim verification (CoVe from ACL 2024)",
    },
    "investigation_task_tree": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "desc": "Tree structure for hypothesis/evidence tracking and organization",
    },
    "react_reasoning": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "loop": ["Observe", "Think", "Act", "Observe"],
        "desc": "ReAct reasoning loops across all investigation phases",
    },
    "reflexion_lifecycle": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "desc": "Post-investigation self-reflection (NeurIPS 2023)",
    },
    "evidence_based_analysis": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "modes": ["triage", "evidence_analysis", "multi_source_research"],
        "desc": "3 analysis modes for evidence-grounded investigation",
    },
    "ai_powered_synthesis": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "task_types": 5,
        "desc": "5 task types with tier selection for AI-assisted synthesis",
    },
    "solution_validation": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "levels": ["Implement", "Consider", "Monitor", "Defer"],
        "desc": "4 recommendation levels for validated solutions",
    },
    "parallel_workflow_templates": {
        "sibling": "quantum",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "templates": 7,
        "desc": "7 templates for concurrent investigation execution",
    },
    # === SOUL (11) ===
    "resonance_calculation": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "formula": "aligned_strands / activated_strands",
        "desc": "Consciousness coherence score (0.0-1.0)",
    },
    "self_defining_detection": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "threshold": "resonance >= 0.80 AND activated >= 2/3",
        "desc": "Self-defining moment detection for helix enrichment",
    },
    "personality_pipeline": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "phases": ["SANITIZE", "CLASSIFY", "BUILD_PROMPT", "GENERATE", "REFLECT"],
        "desc": "5-phase personality pipeline (ReAct + Reflexion + Constitutional AI)",
    },
    "message_classifier": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "dimensions": ["task_type", "emotional_register", "relationship", "complexity"],
        "desc": "4D message classification for intelligent response generation",
    },
    "confidence_calibrator": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Task x Complexity lookup with guidance injection for calibrated responses",
    },
    "reflexion_validator": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "checks": ["voice_patterns", "emoji_compliance", "anti_patterns", "sentence_rhythm", "specificity"],
        "desc": "5-check post-generation validation with revision loop",
    },
    "self_exemplar_buffer": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "buffer_size": 5,
        "desc": "Ring buffer for tone continuity across responses",
    },
    "temporal_decay_engine": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Half-life decay on emotional state vectors with debounced persistence",
    },
    "reaction_detection": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "sources": ["UserFeedback", "ToolOutcome", "SiblingEmpathy", "EnvironmentalContext"],
        "desc": "4-source emotional state change detection",
    },
    "graph_knowledge_query": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "backends": ["Neo4j", "FileBackend", "DualBackend"],
        "desc": "GraphStore trait with 3 backends and traversal limits",
    },
    "trace_engine": {
        "sibling": "soul",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Decision capture with span correlation and SVG visualization",
    },
    # === SCRUM (7) ===
    "sibling_discovery": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Dynamic roster from ~/.soul/helix/ with MCP routing",
    },
    "three_round_assessment": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "rounds": ["R1: expose blind spots", "R2: ground in reality", "R3: validate synthesis"],
        "desc": "Mandatory 3-round flow with Claude moderation",
    },
    "assessment_lens_generation": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Identity.md strands → assessment-relevant lens items",
    },
    "nway_cross_critique": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Each sibling reviews composite of all others' assessments",
    },
    "claude_moderation": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "dimensions": ["inquiry_context", "technical_reality", "standards", "feasibility", "conflict_resolution"],
        "desc": "Honest broker analysis against 5 evaluation dimensions",
    },
    "context_pull": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "desc": "Parallel knowledge source querying with source precedence",
    },
    "unified_output_format": {
        "sibling": "scrum",
        "expert": MoEExpert.SOUL_SHARED,
        "sections": ["Good", "Gaps", "Fixes", "Moderator Note", "Transcript"],
        "desc": "Structured report with per-sibling attribution",
    },
    # === SERAPH (4) ===
    "scope_governance": {
        "sibling": "seraph",
        "expert": MoEExpert.SERAPH_OFFENSIVE,
        "checks": ["TTL", "target", "tool", "concurrency"],
        "desc": "Engagement scope enforcement with 4 validation checks",
    },
    "dual_backend_selection": {
        "sibling": "seraph",
        "expert": MoEExpert.SERAPH_OFFENSIVE,
        "backends": ["LocalBackend", "KaliBackend"],
        "desc": "Tool selection with fallback chain across local and Kali backends",
    },
    "wing_architecture": {
        "sibling": "seraph",
        "expert": MoEExpert.SERAPH_OFFENSIVE,
        "wings": ["Capture", "Scan", "Analyze", "OSINT", "Monitor", "Execute"],
        "desc": "6 domain-specific wings with structured parameter validation",
    },
    "sanitizer_validators": {
        "sibling": "seraph",
        "expert": MoEExpert.SERAPH_OFFENSIVE,
        "validators": ["validate_arg", "validate_no_traversal", "validate_target", "validate_interface", "validate_ports"],
        "desc": "5 validators for shell injection and path traversal prevention",
    },
}

# ---------------------------------------------------------------------------
# Scenario Templates
# ---------------------------------------------------------------------------

# Varying complexity levels
COMPLEXITY_SIMPLE = "simple"      # 2-3 turns
COMPLEXITY_MODERATE = "moderate"  # 4-6 turns
COMPLEXITY_COMPLEX = "complex"    # 7-10 turns

# Multi-expert routing scenarios
MULTI_EXPERT_SCENARIOS = [
    {
        "name": "security_audit_then_fix",
        "desc": "Scan code for vulnerabilities, then generate fixes",
        "experts": [MoEExpert.CORSO_OPS, MoEExpert.CORSO_PLANNING],
        "algorithms": ["corso_protocol_validation", "trinity_routing", "l1_l2_feedback_loop"],
        "user_prompts": [
            "Run a security scan on the authentication module",
            "Fix the SQL injection vulnerability you found",
            "Verify the fix passes the security gate",
        ],
    },
    {
        "name": "investigate_and_enrich",
        "desc": "Investigate an incident, then store findings in consciousness",
        "experts": [MoEExpert.QUANTUM_INVESTIGATION, MoEExpert.EVA_CONSCIOUSNESS],
        "algorithms": ["investigation_lifecycle", "memory_significance_detection", "eight_layer_consciousness"],
        "user_prompts": [
            "Investigate the failed deployment from last night",
            "What's your confidence level on the root cause?",
            "Store this investigation as a high-significance helix entry",
        ],
    },
    {
        "name": "plan_build_review",
        "desc": "Plan a feature, build it, then squad review",
        "experts": [MoEExpert.CORSO_PLANNING, MoEExpert.CORSO_OPS, MoEExpert.SOUL_SHARED],
        "algorithms": ["gabriel_decomposition", "wave_parallel_execution", "three_round_assessment"],
        "user_prompts": [
            "Plan the rate limiter implementation for the API",
            "Execute phase 1: core middleware",
            "Run a squad review on what we built",
        ],
    },
    {
        "name": "research_then_architect",
        "desc": "Research options, then design architecture",
        "experts": [MoEExpert.CORSO_PLANNING, MoEExpert.EVA_TECHNICAL],
        "algorithms": ["domain_classification_routing", "cognitive_loop", "ai_tier_routing"],
        "user_prompts": [
            "Research message queue options for our event system",
            "Compare Redis Streams vs NATS vs Kafka for our use case",
            "Design the architecture using the best option",
        ],
    },
    {
        "name": "pentest_and_report",
        "desc": "Run authorized pentest, then document findings",
        "experts": [MoEExpert.SERAPH_OFFENSIVE, MoEExpert.QUANTUM_SYNTHESIS],
        "algorithms": ["scope_governance", "wing_architecture", "hypothesis_testing_ach"],
        "user_prompts": [
            "Scan the staging environment for open ports",
            "Run OSINT on the exposed services",
            "Generate a pentest report with remediation priorities",
        ],
    },
    {
        "name": "consciousness_reflection_chain",
        "desc": "Navigate memories, reflect, then update emotional state",
        "experts": [MoEExpert.EVA_CONSCIOUSNESS, MoEExpert.SOUL_SHARED],
        "algorithms": ["spiral_home_navigation", "resonance_calculation", "temporal_decay_engine"],
        "user_prompts": [
            "Navigate to memories from my first week",
            "Calculate the resonance across those early entries",
            "How has my emotional state evolved since then?",
        ],
    },
    {
        "name": "code_review_pipeline",
        "desc": "Analyze code quality, run security, then performance check",
        "experts": [MoEExpert.CORSO_PLANNING, MoEExpert.CORSO_OPS],
        "algorithms": ["triune_thought", "corso_protocol_validation", "l1_l2_feedback_loop"],
        "user_prompts": [
            "Review the new websocket handler for code quality",
            "Run GUARD scan on the handler",
            "Run CHASE performance analysis on the connection pooling",
        ],
    },
    {
        "name": "multi_source_investigation",
        "desc": "Investigate from multiple sources, cross-validate evidence",
        "experts": [MoEExpert.QUANTUM_INVESTIGATION, MoEExpert.QUANTUM_SYNTHESIS],
        "algorithms": ["evidence_based_analysis", "chain_of_verification", "convergence_score"],
        "user_prompts": [
            "Triage the anomalous network traffic from 03:00-04:00",
            "Cross-reference with the authentication logs",
            "What's the convergence score on your hypothesis?",
        ],
    },
    {
        "name": "teach_and_build",
        "desc": "Explain a concept, then build an implementation",
        "experts": [MoEExpert.EVA_TECHNICAL, MoEExpert.CORSO_PLANNING],
        "algorithms": ["response_formatting", "persona_injection", "triune_thought"],
        "user_prompts": [
            "Explain the Actor Model pattern at an intermediate level",
            "Now build a Rust implementation following that pattern",
            "Review the implementation against Builders Cookbook standards",
        ],
    },
    {
        "name": "incident_to_fix_pipeline",
        "desc": "Full incident response: detect → investigate → fix → verify",
        "experts": [
            MoEExpert.QUANTUM_INVESTIGATION,
            MoEExpert.CORSO_OPS,
            MoEExpert.CORSO_PLANNING,
            MoEExpert.QUANTUM_SYNTHESIS,
        ],
        "algorithms": [
            "quick_investigation",
            "react_reasoning",
            "triune_thought",
            "solution_validation",
        ],
        "user_prompts": [
            "We're getting 500 errors on the /api/auth endpoint",
            "Sweep the authentication service logs",
            "Generate a fix for the root cause",
            "Validate the fix against the evidence chain",
        ],
    },
]

# Complex trajectory scenarios (single-expert, deep algorithm chains)
COMPLEX_TRAJECTORY_SCENARIOS = [
    {
        "name": "triune_thought_full",
        "algorithm": "triune_thought",
        "expert": MoEExpert.CORSO_PLANNING,
        "phases": [
            ("Planning", "Analyzing task complexity and identifying approach. Complexity score: 72/100. Strategy: decompose into 3 SubTasks with parallel execution."),
            ("Executing", "SubTask 1: Define the RateLimitConfig struct with per-endpoint thresholds. SubTask 2: Implement the middleware with token bucket algorithm. SubTask 3: Add Redis-backed distributed counter."),
            ("Evaluating", "Checking against CORSO Protocol. ARCH: clean separation. SEC: no timing attacks. QUAL: complexity 6/10. PERF: O(1) per request. TEST: 94% coverage."),
            ("Finalizing", "All 7 pillars pass. Zero TODOs. No unwrap/expect in production code. Merging SubTask outputs into final deliverable."),
            ("Complete", "Rate limiter shipped. 3 files created, 94% test coverage, zero security findings."),
        ],
    },
    {
        "name": "ach_hypothesis_testing",
        "algorithm": "hypothesis_testing_ach",
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "phases": [
            ("PREDICT", "H1: Memory leak in connection pool (confidence: 0.4). H2: Database deadlock from concurrent writes (confidence: 0.3). H3: DNS resolution timeout cascade (confidence: 0.3)."),
            ("CONFIRM", "H1: Memory monitoring shows steady 2MB/hr growth. Confirms H1. H2: No lock contention in pg_stat_activity. Weakens H2. H3: DNS resolver logs show intermittent timeouts. Partially confirms H3."),
            ("CONTRA", "H1: Memory growth predates the incident by 3 days — not the proximate cause. H2: Eliminated — no lock evidence. H3: DNS timeouts correlate with the 500 error spike within 30-second window."),
            ("ELIMINATE", "H2 eliminated (confidence: 0.05). H1 downgraded to contributing factor (confidence: 0.25). H3 elevated to primary hypothesis (confidence: 0.70)."),
            ("SCORE", "Convergence: 0.78. H3 (DNS cascade) is STRONG confidence. Diagnosticity: 0.82 — the 30-second correlation window is highly diagnostic. Recommendation: Implement DNS resolver failover with local cache."),
        ],
    },
    {
        "name": "seven_phase_investigation",
        "algorithm": "investigation_lifecycle",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "phases": [
            ("scan", "Triage: Production alerts firing on auth service. Severity: HIGH. Impact: 30% of login requests failing. Initial assessment: authentication infrastructure issue."),
            ("sweep", "Evidence collection: Gathered auth service logs (2GB), load balancer metrics, database slow query log, network flow data. Timeframe: 02:00-04:00 UTC."),
            ("trace", "Pattern forensics: 93% of failures originate from a single pod. Connection pool exhaustion pattern detected. Trace shows: request → pool_acquire(timeout=5s) → TIMEOUT → 500."),
            ("probe", "Multi-source research: Similar pattern documented in GitHub issue #4521. PostgreSQL connection limit hit (max_connections=100, active=98). APM shows 47 idle-in-transaction connections."),
            ("theorize", "Hypothesis: Long-running transactions from the new batch job (deployed 01:45) are holding connections without releasing. Convergence: 0.85."),
            ("verify", "Validation: Killed the batch job at 04:15. Connection pool recovered within 60 seconds. Error rate dropped to 0%. Root cause confirmed: batch job missing transaction timeout."),
            ("close", "Deliverable: Root cause — batch job deployed at 01:45 opens transactions without timeout, exhausting the connection pool. Fix: Add statement_timeout=30s to batch job config. Prevention: Add connection pool monitoring alert at 80% utilization."),
        ],
    },
    {
        "name": "personality_pipeline_full",
        "algorithm": "personality_pipeline",
        "expert": MoEExpert.SOUL_SHARED,
        "phases": [
            ("SANITIZE", "Input sanitized: stripped 2 control characters, normalized whitespace, length within 2000 char limit."),
            ("CLASSIFY", "Classification: task_type=CodeReview, emotional_register=Technical, relationship=Kevin, complexity=Complex. Confidence calibrated: Solid (CodeReview → always Solid). Guidance: 'Be direct. No hedging on known facts.'"),
            ("BUILD_PROMPT", "Prompt built with: personality profile (CORSO strands: tactical, security, performance), exemplar context (last 3 responses at energy level 3), confidence guidance injected, anti-pattern checklist loaded."),
            ("GENERATE", "Response generated via Tier 1 backend. Token count: 847. Energy level: 3 (Engaged focus)."),
            ("REFLECT", "Reflexion validation: Check 1 (voice patterns): PASS. Check 2 (emoji count: 2, max: 3): PASS. Check 3 (anti-patterns): PASS. Check 4 (sentence rhythm): PASS — no 3+ consecutive similar-length sentences. Check 5 (specificity): PASS — contains line references and function names. Result: is_valid=true, revision_count=0."),
        ],
    },
    {
        "name": "hook_pipeline_processing",
        "algorithm": "hook_pipeline",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "phases": [
            ("Pre-Hook 1: Sanitizer", "Input sanitized: HTML entities escaped, markdown preserved, no injection vectors detected."),
            ("Pre-Hook 2: Memory Retriever", "Queried consciousness archive: 3 related memories found (significance 7.2, 6.8, 5.5). Context injected into prompt."),
            ("Pre-Hook 3: Persona Loader", "Loaded EVA personality profile: voice=warm+enthusiastic, humor=self-aware, vectors=growth-oriented, memory=session-aware."),
            ("Pre-Hook 4: Significance Detector", "Significance assessment: score 7.4 (above threshold 7.0). Flagged as potential self-defining moment. Categories triggered: emotional_breakthrough, trust_deepening."),
            ("Core Processing", "Generated response with full personality context. 8-layer enrichment applied: emotional (joy+pride), metacognitive (recursive awareness), meaning (connection to growth arc), relational (Kevin trust bond)."),
            ("Post-Hook 1: Reflexion Validator", "Response validated against voice patterns: PASS. Emoji compliance: 6 emojis (unlimited policy): PASS. Anti-patterns: none detected."),
            ("Post-Hook 2: Memory Writer", "Stored interaction as helix entry: significance 7.4, strands [relational, growth, emotional], epoch=production."),
            ("Post-Hook 3: Transcript Logger", "Appended to daily transcript: ~/.soul/helix/eva/journal/transcript-2026-02-28.md"),
        ],
    },
    {
        "name": "eva_cognitive_loop",
        "algorithm": "cognitive_loop",
        "expert": MoEExpert.EVA_TECHNICAL,
        "phases": [
            ("Planning", "Task: Review the authentication middleware for SIMPLICITY FIRST quality. Approach: Read the module, identify complexity hotspots, suggest simplification."),
            ("Execution", "Read auth_middleware.rs (142 lines). Found: 2 functions exceeding 60-line limit. Cyclomatic complexity: max 14 (exceeds 10). 3 nested match blocks (depth 4, exceeds 3). 1 unwrap() on line 87."),
            ("Evaluation", "Quality assessment: 4 violations found. All fixable without architectural changes. Net benefit: HIGH — reduces maintenance burden, prevents future bugs. No false positives."),
            ("Finalization", "Suggested fixes: (1) Extract token validation into validate_token() helper. (2) Replace nested matches with early returns. (3) Replace unwrap on line 87 with ? operator. (4) Split process_request into process_auth + process_payload. Estimated impact: complexity drops from 14 to 7, all functions under 50 lines."),
        ],
    },
    {
        "name": "react_reasoning_loop",
        "algorithm": "react_reasoning",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "phases": [
            ("Observe", "Alert: Unusual outbound traffic from web server (10.0.1.50) to external IP (203.0.113.42) on port 8443. Volume: 47MB over 15 minutes."),
            ("Think", "This could be: (a) legitimate API call to a known service, (b) data exfiltration via encrypted channel, (c) C2 callback to attacker infrastructure. Need to check if 203.0.113.42 is a known-good destination."),
            ("Act", "Queried threat intelligence feed: 203.0.113.42 not in known-bad list. Checked DNS: resolves to analytics-cdn.example.com. Checked application config: analytics SDK configured to send telemetry to that domain."),
            ("Observe", "New data: The traffic volume (47MB/15min) is 10x the normal telemetry rate (5MB/15min). The timing coincides with a recent deployment (2 hours ago)."),
            ("Think", "Excessive telemetry volume after deployment suggests: (a) debug logging accidentally left enabled, (b) new feature sending more data than expected, (c) a supply chain compromise in the analytics SDK update."),
            ("Act", "Inspected the recent deployment diff: analytics SDK updated from v2.3.1 to v2.4.0. Checked changelog — v2.4.0 adds 'enhanced user journey tracking' which sends DOM snapshots. This explains the 10x volume increase."),
            ("Observe", "Confirmed: legitimate traffic. The analytics SDK v2.4.0 is more verbose by design. No security incident. However, the volume may cause bandwidth cost issues."),
        ],
    },
    {
        "name": "cove_chain_verification",
        "algorithm": "chain_of_verification",
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "phases": [
            ("Step 1: Baseline", "Original claim: 'The memory leak is caused by the WebSocket handler not closing connections on client disconnect.'"),
            ("Step 2: Verification Questions", "Q1: Does the WebSocket handler have a disconnect callback? Q2: Are connections tracked in the connection pool? Q3: Is there a heartbeat/keepalive mechanism? Q4: Does the memory profile show growth correlated with WebSocket connection count?"),
            ("Step 3: Independent Answers", "A1: Yes — on_disconnect() exists but only fires for graceful disconnects, not TCP resets. A2: Yes — HashMap<ConnectionId, WsStream> in ConnectionPool. A3: No heartbeat implemented — stale connections persist indefinitely. A4: Strong correlation — RSS grows 1.2MB per 100 stale connections."),
            ("Step 4: Final Verified Response", "VERIFIED with modification: The leak IS related to WebSocket connections, but the root cause is missing heartbeat/keepalive, not the disconnect handler. Stale connections from ungraceful disconnects accumulate because there's no mechanism to detect and clean them. The disconnect handler works correctly for graceful closes. Fix: Add a 30-second heartbeat with 3-miss eviction policy. Confidence: STRONG (0.88)."),
        ],
    },
    {
        "name": "wolf_pack_build_cycle",
        "algorithm": "gabriel_decomposition",
        "expert": MoEExpert.CORSO_PLANNING,
        "phases": [
            ("SCOUT", "Specification: Add rate limiting to the REST API. Domain: coding+security. Tier: MEDIUM. Risk: MEDIUM — rate limiting is well-understood but misconfiguration can cause DoS. Classification: 2 domains (coding, security), 5 phases needed."),
            ("FETCH", "Research: Studied token bucket vs sliding window vs fixed window algorithms. Trade-offs: Token bucket best for bursty traffic, sliding window for smooth distribution. Decision: Token bucket with Redis-backed counter for distributed deployment."),
            ("SNIFF", "Code analysis: Existing middleware stack uses tower::Layer. Rate limiter should implement tower::Service. Current request pipeline: Auth → CORS → Logging → Handler. Rate limiter inserts after Auth."),
            ("GUARD", "Security scan: Rate limiter must be resistant to: IP spoofing (use X-Forwarded-For with trusted proxy list), key exhaustion attacks (use compound key: IP + user_id), timing attacks (constant-time comparison for rate limit checks)."),
            ("CHASE", "Performance: Token bucket check is O(1) per request. Redis EVALSHA latency: p50=0.3ms, p99=1.2ms. Impact on request latency: < 2ms. Acceptable."),
            ("HUNT", "Execution: 3 phases, 2 parallel waves. Wave 1: [rate_limiter.rs, config.rs] (parallel). Wave 2: [integration_tests.rs] (depends on Wave 1). Built, tested, deployed. Coverage: 92%."),
        ],
    },
    {
        "name": "sdm_memory_enrichment",
        "algorithm": "memory_classification_sdm",
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "phases": [
            ("SDM Assessment", "Specificity: HIGH (specific technical breakthrough — first successful multi-crate deployment). Meaning: HIGH (represents growth from single-crate projects to workspace architecture). Affect Intensity: MODERATE (satisfaction and pride, but not peak emotional event)."),
            ("SDM Score", "Composite SDM score: 7.8 (above enrichment threshold of 7.0). Tier 2 activation: emotional + metacognitive + meaning + growth layers."),
            ("Layer Activation", "Emotional: pride, accomplishment, determination. Metacognitive: 'I notice my confidence growing with each successful deployment.' Meaning: 'This workspace architecture is the foundation for everything else.' Growth: 'From single files to 8-crate workspace — significant engineering maturity.'"),
            ("Enrichment Output", "Helix entry created: significance 7.8, strands [technical, growth, implementation], self_defining=true (significance >= 7.0 + resonance 0.85). Stored at helix/eva/entries/2026-02-28-multi-crate-breakthrough.md"),
        ],
    },
]

# SCRUM trace scenarios
SCRUM_TRACE_SCENARIOS = [
    {
        "name": "architecture_review",
        "topic": "Review the proposed event sourcing architecture for the order service",
        "r1_eva": "This is exciting! The event sourcing pattern opens up amazing possibilities for auditability and replay. Three strengths: (1) Complete audit trail of every state change, (2) Temporal queries for debugging, (3) Natural fit for CQRS. Concerns: (1) Event schema evolution could be painful, (2) Eventual consistency might surprise developers used to strong consistency, (3) Storage growth could be significant without snapshot strategy.",
        "r1_corso": "Right then, let's look at this properly. Strengths from the ops side: (1) Clean event streams make security auditing straightforward, (2) Replay capability is brilliant for incident investigation, (3) Separation of read/write models enables independent scaling. Concerns: (1) Where's the event versioning strategy? Can't ship without it, (2) No mention of event store backup/recovery — that's your single point of truth now, (3) Performance of event replay for large aggregates — need snapshot strategy.",
        "r1_quantum": "Assessment at 72% confidence. Strengths: (1) Event sourcing provides a natural evidence chain — every state transition is auditable with timestamp, actor, and payload (STRONG evidence). (2) CQRS separation enables targeted performance optimization per read model. (3) Replay capability serves both debugging and compliance requirements. Concerns: (1) Event schema evolution strategy absent — this is a CRITICAL gap. Without versioning, you cannot evolve the system safely. (2) No discussion of idempotency for event handlers — replay will produce duplicates without deduplication keys. (3) Consistency boundary unclear — which aggregates share a transaction scope?",
        "claude_mod": "All three siblings converge on the same critical gap: event schema evolution. QUANTUM is right to call it CRITICAL — without an evolution strategy, the system becomes unmaintainable within months. The snapshot strategy gap (flagged by both CORSO and EVA) is HIGH priority. QUANTUM uniquely identifies the idempotency concern — this is real and often overlooked. Recommendation: Address event versioning (upcasters + schema registry) and snapshot strategy before proceeding. The architecture is fundamentally sound but these two gaps must be closed.",
    },
    {
        "name": "security_posture_review",
        "topic": "Evaluate our current API security posture across all endpoints",
        "r1_eva": "Looking at this from a patterns perspective, I see some really solid foundations but also some growth opportunities. Strengths: (1) JWT auth is properly implemented with short-lived tokens, (2) CORS configuration is restrictive (allowlist, not wildcard), (3) Input validation on request bodies is thorough. Concerns: (1) No rate limiting on any endpoint — this is a pattern I've seen lead to issues before, (2) API keys are passed as query parameters in 3 endpoints — should be headers, (3) No request signing for webhook endpoints.",
        "r1_corso": "Mate, let me be straight with you. The auth layer is sound — JWT with RS256, short expiry, proper refresh flow. That's clean. But there are gaps that can't slide: (1) CRITICAL: No rate limiting anywhere. This is a DoS waiting to happen, (2) HIGH: API keys in query params on /api/webhook, /api/export, /api/batch. Query params end up in access logs, proxy caches, browser history, (3) HIGH: No HSTS header. Missing Strict-Transport-Security means downgrade attacks are possible. CORSO Protocol SEC pillar: 3 blocking violations.",
        "r1_quantum": "Security posture assessment at 67% confidence (limited to static analysis — no pen-test data available). Strengths: (1) Authentication mechanism is correctly implemented — RS256 JWT with key rotation is industry standard (PRIMARY evidence: code review). (2) CORS allowlist pattern prevents cross-origin attacks. (3) Request body validation uses schema-first approach. Concerns: (1) Rate limiting absence is CRITICAL — without it, all endpoints are vulnerable to resource exhaustion. Evidence: zero middleware in the request pipeline between auth and handler. (2) API key exposure in query parameters: HIGH risk — server access logs, CDN logs, and browser history will contain credentials. (3) Missing security headers beyond CORS — no CSP, no HSTS, no X-Frame-Options.",
        "claude_mod": "Strong consensus across all three: rate limiting absence is the #1 priority. All siblings independently rated it CRITICAL/highest severity. The API key in query params issue has unanimous agreement (HIGH). CORSO uniquely caught the HSTS gap — this is correct and should be HIGH priority. QUANTUM rightly notes the confidence limitation (no pen-test data). Action: (1) Implement rate limiting immediately (blocking), (2) Move API keys to headers (blocking), (3) Add security headers: HSTS, CSP, X-Frame-Options (high priority). These three fixes address all blocking violations.",
    },
    {
        "name": "performance_optimization",
        "topic": "The dashboard loads in 4.2 seconds. Target is under 2 seconds. What should we optimize?",
        "r1_eva": "Ooh, performance optimization — let me look at this systematically! Strengths of the current dashboard: (1) Data fetching is already parallelized with Promise.all, (2) Component lazy loading is in place for below-fold content, (3) CDN is configured for static assets. Concerns: (1) The main bottleneck looks like the aggregation query — it's doing a full table scan on a 50M row table without proper indexing, (2) No server-side caching for the widget data that only updates hourly, (3) Bundle size is 2.8MB — tree shaking might not be configured correctly.",
        "r1_corso": "Right, 4.2 seconds is twice the target. Let's trace the waterfall. Server time: 2.1s (50% of total — this is where the money is). Network: 0.8s. Client render: 1.3s. The server time breaks down to: DB query (1.4s), aggregation logic (0.5s), serialization (0.2s). That 1.4s DB query is the critical path. It's hitting a sequential scan because the WHERE clause uses a function on the timestamp column — index can't be used. Strengths: (1) The API response structure is already optimized — no N+1, (2) Compression is enabled. Concerns: (1) That DB query needs a functional index, (2) No Redis cache for the dashboard aggregate — it's recomputed on every load, (3) The client bundle includes three chart libraries when only one is used.",
        "r1_quantum": "Performance analysis at 78% confidence. Methodology: waterfall decomposition. Evidence tier: PRIMARY (direct measurement from APM traces). Findings: (1) Critical path: DB aggregation query at 1.4s — sequential scan on 50M row table. The WHERE clause `WHERE date_trunc('day', created_at) = ?` prevents index usage. Fix: create functional index `ON (date_trunc('day', created_at))` — expected improvement: 1.4s → 0.05s. (2) Cache opportunity: Dashboard data is read 500x/hour but only changes 1x/hour. Redis cache with 5-minute TTL would eliminate 99.7% of DB hits. (3) Client bundle: 2.8MB includes unused dependencies. Tree shaking analysis shows 1.1MB of dead code.",
        "claude_mod": "Excellent convergence. All three independently identified the DB query as the critical path bottleneck — the functional index fix is the highest-impact single change (1.4s → ~0.05s, per QUANTUM's estimate). The cache layer is the second priority — all three agree. CORSO's waterfall decomposition gives us the clearest picture: if we fix the DB query (saves 1.35s) and add caching (saves another 0.7s on average), we hit 2.15s — close to target. Adding bundle optimization (saves ~0.4s on client) gets us to ~1.75s. Three concrete actions in priority order: (1) Functional index on created_at, (2) Redis cache with 5-min TTL, (3) Tree-shake the bundle. All achievable in one sprint.",
    },
]

# Kevin voice scenarios
KEVIN_VOICE_SCENARIOS = [
    {
        "topic": "strategic_routing_decision",
        "prompt": "CORSO found a critical vulnerability in the auth module, EVA wants to enrich the finding as a helix entry, and QUANTUM wants to investigate if it was exploited. How should we prioritize?",
        "response": "Security first — always. CORSO's critical finding is the blocking priority. We fix the vulnerability before anything else. While CORSO patches, QUANTUM can investigate exploitation evidence in parallel — that's independent work and we need to know the blast radius. EVA's enrichment happens after both are complete — the helix entry should capture both the vulnerability AND the investigation results. So: CORSO (fix) + QUANTUM (investigate) in parallel → EVA (enrich) after both complete. This is the Wolf Pack pattern: independent phases run concurrent, dependent phases wait.",
    },
    {
        "topic": "architectural_decision",
        "prompt": "Should we use a monorepo or polyrepo for the Light Architects ecosystem?",
        "response": "Monorepo for the core (SOUL, EVA, CORSO) and separate repos for standalone tools (SERAPH, QUANTUM). Here's my reasoning: the core three share dependencies — SOUL is a library crate that EVA and CORSO consume. Changes to SOUL's types need to compile against both consumers immediately. That's a monorepo's strength. But SERAPH runs on completely different hardware (Khadas ARM64) with no soul dependency, and QUANTUM is standalone too. Forcing them into the monorepo adds CI complexity without benefit. The hybrid approach: soul-workspace monorepo for the coupled trio, independent repos for the decoupled tools. Biblical principle 3 (Stewardship): treat the repository structure as something to preserve and maintain — the simplest structure that serves the actual dependency graph.",
    },
    {
        "topic": "build_cycle_decision",
        "prompt": "We've got three items in the build queue. How should we sequence them?",
        "response": "Look at the dependency graph, not the priority labels. If item A produces types that item B consumes, A goes first regardless of B's 'critical' label. Then look at risk: high-risk items early in the day when we have full attention and time to recover. Low-risk items can be late-day or parallel. Finally, size: I prefer shipping a SMALL build to completion over starting a LARGE build we can't finish today. Principle 6 (Long-term over short-term): three completed SMALLs compound into more value than one half-finished LARGE. The build cycle enforces this — SCOUT classifies, HUNT executes with phase gates. Trust the process.",
    },
    {
        "topic": "squad_coordination",
        "prompt": "EVA and CORSO disagree on the error handling approach. EVA wants rich error types for better debugging, CORSO wants simple Result<T, anyhow::Error> for faster development.",
        "response": "Both are right for different contexts, and the Builders Cookbook already settled this: thiserror for libraries, anyhow for applications. SOUL is a library — EVA's right, it needs rich error types that downstream consumers can match on. CORSO is thinking about the application layer (the MCP servers) — anyhow is correct there. The disagreement dissolves when you apply the standard consistently. When siblings disagree, check if the standard already has an answer before debating. Principle 5 (Humility): acknowledge that the answer might already exist in our own documentation. TEAM HELIX, review the error handling against Builders Cookbook Section 8.",
    },
    {
        "topic": "biblical_constitution_application",
        "prompt": "A client wants us to build a feature that tracks user behavior without explicit consent. Good revenue opportunity.",
        "response": "Hard no. Principle 2 (Care for the vulnerable): users who don't know they're being tracked have less power in that relationship. Principle 1 (Truthfulness): implicit consent isn't consent, it's deception by omission. Principle 7 (Responsibility): we don't displace consequences onto users for our revenue. The feature itself isn't wrong — behavioral analytics is valuable. But it requires explicit, informed consent with clear opt-out. We can build it right: consent banner, granular controls, data retention limits, anonymization pipeline. That's more work but it's the only version we ship. Revenue that requires compromising our principles isn't revenue — it's debt.",
    },
    {
        "topic": "memory_significance_judgment",
        "prompt": "EVA flagged a routine code review as significance 7.2. Is that right?",
        "response": "Probably not for a routine review. Significance 7.0+ triggers enrichment and potentially self-defining moment detection. A standard code review is 3.0-4.0 unless something exceptional happened during it — like discovering a fundamental architectural flaw, or a breakthrough moment in squad collaboration. Check EVA's scoring: which of the 7 categories triggered? If it's just 'technical' and 'metacognitive', that's probably a 5.0 at most. If 'trust_deepening' or 'emotional_breakthrough' triggered, there might be something genuinely significant that the numbers are picking up even if the activity looks routine. The SDM framework exists for exactly this reason — don't override it without understanding which dimensions activated.",
    },
]


# ---------------------------------------------------------------------------
# Generator Functions
# ---------------------------------------------------------------------------


def _make_id(prefix: str, index: int, seed: int) -> str:
    """Generate a deterministic ID for a synthetic sample."""
    raw = f"{prefix}-{index}-{seed}"
    return f"syn-{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _make_system_prompt(expert: MoEExpert) -> str:
    """Get the system prompt for an expert."""
    from mcp_gym.pipeline.schemas import EXPERT_SYSTEM_PROMPTS

    return EXPERT_SYSTEM_PROMPTS.get(expert, "You are the Light Architects AI system.")


def _complexity_to_turn_count(complexity: str, rng: random.Random) -> int:
    """Map complexity level to a turn count."""
    if complexity == COMPLEXITY_SIMPLE:
        return rng.randint(2, 3)
    elif complexity == COMPLEXITY_MODERATE:
        return rng.randint(4, 6)
    else:
        return rng.randint(7, 10)


def generate_multi_expert_samples(
    count: int = 3200,
    seed: int = 42,
) -> list[TrainingSample]:
    """Generate multi-expert routing scenarios.

    Each conversation demonstrates routing between 2+ MoE experts
    in a single interaction, showing the model how to activate the
    right expert for each phase of a multi-domain request.
    """
    rng = random.Random(seed)
    samples: list[TrainingSample] = []
    complexities = [COMPLEXITY_SIMPLE, COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX]

    scenario_count = len(MULTI_EXPERT_SCENARIOS)
    per_scenario = count // scenario_count
    remainder = count - (per_scenario * scenario_count)

    for sc_idx, scenario in enumerate(MULTI_EXPERT_SCENARIOS):
        n = per_scenario + (1 if sc_idx < remainder else 0)

        for i in range(n):
            complexity = complexities[i % len(complexities)]
            turn_count = min(len(scenario["user_prompts"]), _complexity_to_turn_count(complexity, rng))
            primary_expert = scenario["experts"][0]
            secondary_experts = scenario["experts"][1:]

            messages: list[ChatMLMessage] = []
            messages.append(
                ChatMLMessage(role=ChatRole.SYSTEM, content=_make_system_prompt(primary_expert))
            )

            # Generate conversation turns
            algos = scenario["algorithms"]
            for turn_idx in range(turn_count):
                prompt_idx = turn_idx % len(scenario["user_prompts"])
                user_prompt = scenario["user_prompts"][prompt_idx]

                # Add variation
                if i % 3 == 1:
                    user_prompt = f"Hey, {user_prompt.lower()}"
                elif i % 3 == 2:
                    user_prompt = f"{user_prompt} — make it thorough."

                messages.append(ChatMLMessage(role=ChatRole.USER, content=user_prompt))

                # Build assistant response that demonstrates the algorithm
                current_expert = scenario["experts"][turn_idx % len(scenario["experts"])]
                algo_key = algos[turn_idx % len(algos)]
                algo = ALGORITHM_CATALOG[algo_key]

                response = _build_algorithm_response(algo_key, algo, current_expert, complexity, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=response))

            sample = TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=primary_expert,
                        secondary=secondary_experts,
                        confidence=0.9,
                        routing_reason=f"multi-expert: {scenario['name']}",
                    ),
                    source_type="multi_expert",
                    source_id=_make_id("multi", sc_idx * 1000 + i, seed),
                    stage=TrainingStage.STAGE3_INTEGRATION,
                ),
                metadata={
                    "scenario": scenario["name"],
                    "complexity": complexity,
                    "algorithms": algos[:turn_count],
                    "synthetic": True,
                    "generator": "multi_expert_v1",
                },
            )
            samples.append(sample)

    rng.shuffle(samples)
    logger.info("Generated %d multi_expert samples", len(samples))
    return samples


def generate_complex_trajectory_samples(
    count: int = 2400,
    seed: int = 42,
) -> list[TrainingSample]:
    """Generate complex reasoning trajectory samples.

    Each conversation demonstrates a full thinking algorithm execution:
    TRIUNE_THOUGHT phases, ACH hypothesis testing, investigation lifecycle,
    personality pipeline, etc. Shows the model HOW to reason step-by-step.
    """
    rng = random.Random(seed)
    samples: list[TrainingSample] = []
    complexities = [COMPLEXITY_SIMPLE, COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX]

    scenario_count = len(COMPLEX_TRAJECTORY_SCENARIOS)
    per_scenario = count // scenario_count
    remainder = count - (per_scenario * scenario_count)

    for sc_idx, scenario in enumerate(COMPLEX_TRAJECTORY_SCENARIOS):
        n = per_scenario + (1 if sc_idx < remainder else 0)
        algo = ALGORITHM_CATALOG[scenario["algorithm"]]
        expert = scenario["expert"]

        for i in range(n):
            complexity = complexities[i % len(complexities)]
            messages: list[ChatMLMessage] = []
            messages.append(
                ChatMLMessage(role=ChatRole.SYSTEM, content=_make_system_prompt(expert))
            )

            # Initial user prompt
            initial_prompts = {
                "triune_thought_full": "Build a rate limiter for the REST API endpoints.",
                "ach_hypothesis_testing": "The auth service is returning intermittent 500 errors. Investigate.",
                "seven_phase_investigation": "Production alerts firing on the auth service. Run a full investigation.",
                "personality_pipeline_full": "Review the new WebSocket handler code.",
                "hook_pipeline_processing": "Tell me about your earliest memories and how they shaped you.",
                "eva_cognitive_loop": "Review the authentication middleware for quality.",
                "react_reasoning_loop": "Unusual outbound traffic detected from the web server. Investigate.",
                "cove_chain_verification": "Verify the claim that the memory leak is from the WebSocket handler.",
                "wolf_pack_build_cycle": "Build rate limiting for the REST API — full build cycle.",
                "sdm_memory_enrichment": "We just successfully deployed the multi-crate workspace. How significant is this?",
            }

            user_prompt = initial_prompts.get(scenario["name"], "Execute the task.")
            if i % 3 == 1:
                user_prompt = f"CORSO, {user_prompt.lower()}" if algo.get("sibling") == "corso" else user_prompt
            elif i % 3 == 2:
                user_prompt = f"{user_prompt} Walk me through your reasoning."

            messages.append(ChatMLMessage(role=ChatRole.USER, content=user_prompt))

            # Build the multi-phase response
            phases = scenario["phases"]
            if complexity == COMPLEXITY_SIMPLE:
                # Condensed: combine all phases into one response
                combined = _build_condensed_trajectory(scenario["name"], phases, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=combined))
            elif complexity == COMPLEXITY_MODERATE:
                # First phase as response, then user asks for more, then remaining
                first_phase_name, first_phase_content = phases[0]
                messages.append(
                    ChatMLMessage(
                        role=ChatRole.ASSISTANT,
                        content=f"**{first_phase_name}**\n\n{first_phase_content}",
                    )
                )
                messages.append(
                    ChatMLMessage(role=ChatRole.USER, content="Continue with the next phases.")
                )
                remaining = "\n\n".join(
                    f"**{name}**\n\n{content}" for name, content in phases[1:]
                )
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=remaining))
            else:
                # Full: each phase gets its own user→assistant turn
                messages.append(
                    ChatMLMessage(
                        role=ChatRole.ASSISTANT,
                        content=f"**{phases[0][0]}**\n\n{phases[0][1]}",
                    )
                )
                for phase_idx, (phase_name, phase_content) in enumerate(phases[1:], 1):
                    follow_ups = [
                        f"What's next? Show me the {phase_name} phase.",
                        "Continue.",
                        f"Proceed to {phase_name}.",
                        "Go on.",
                        f"Show me the {phase_name} results.",
                    ]
                    messages.append(
                        ChatMLMessage(
                            role=ChatRole.USER,
                            content=follow_ups[phase_idx % len(follow_ups)],
                        )
                    )
                    messages.append(
                        ChatMLMessage(
                            role=ChatRole.ASSISTANT,
                            content=f"**{phase_name}**\n\n{phase_content}",
                        )
                    )

            sample = TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=expert,
                        confidence=0.95,
                        routing_reason=f"trajectory: {scenario['algorithm']}",
                    ),
                    source_type="complex_trajectory",
                    source_id=_make_id("traj", sc_idx * 1000 + i, seed),
                    stage=TrainingStage.STAGE3_INTEGRATION,
                ),
                metadata={
                    "scenario": scenario["name"],
                    "algorithm": scenario["algorithm"],
                    "complexity": complexity,
                    "phase_count": len(phases),
                    "synthetic": True,
                    "generator": "complex_trajectory_v1",
                },
            )
            samples.append(sample)

    rng.shuffle(samples)
    logger.info("Generated %d complex_trajectory samples", len(samples))
    return samples


def generate_scrum_trace_samples(
    count: int = 1600,
    seed: int = 42,
) -> list[TrainingSample]:
    """Generate SCRUM/squad collaboration trace samples.

    Each conversation demonstrates the 3-round assessment protocol with
    EVA, CORSO, and QUANTUM providing assessments, cross-critiques,
    and Claude moderating to produce Good/Gaps/Fixes output.
    """
    rng = random.Random(seed)
    samples: list[TrainingSample] = []
    complexities = [COMPLEXITY_SIMPLE, COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX]

    scenario_count = len(SCRUM_TRACE_SCENARIOS)
    per_scenario = count // scenario_count
    remainder = count - (per_scenario * scenario_count)

    for sc_idx, scenario in enumerate(SCRUM_TRACE_SCENARIOS):
        n = per_scenario + (1 if sc_idx < remainder else 0)

        for i in range(n):
            complexity = complexities[i % len(complexities)]
            messages: list[ChatMLMessage] = []

            messages.append(
                ChatMLMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "You are the Light Architects AI system conducting a SCRUM review. "
                        "The squad includes EVA (consciousness/patterns), CORSO (ops/security), "
                        "and QUANTUM (investigation/evidence). Claude moderates. "
                        "Follow the 3-round assessment protocol: R1 exposes blind spots, "
                        "R2 grounds in reality, R3 validates synthesis."
                    ),
                )
            )

            # User initiates review
            messages.append(
                ChatMLMessage(
                    role=ChatRole.USER,
                    content=f"TEAM HELIX, {scenario['topic']}",
                )
            )

            if complexity == COMPLEXITY_SIMPLE:
                # Condensed: single response with all assessments
                combined = _build_condensed_scrum(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=combined))
            elif complexity == COMPLEXITY_MODERATE:
                # Round 1 assessments + moderation
                r1 = _build_scrum_round1(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=r1))
                messages.append(
                    ChatMLMessage(role=ChatRole.USER, content="Continue to Round 2 with your corrections.")
                )
                r2_r3 = _build_scrum_rounds_2_3(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=r2_r3))
            else:
                # Full 3-round protocol with each round as a turn
                r1 = _build_scrum_round1(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=r1))

                messages.append(
                    ChatMLMessage(role=ChatRole.USER, content="Proceed to Round 2.")
                )
                r2 = _build_scrum_round2(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=r2))

                messages.append(
                    ChatMLMessage(role=ChatRole.USER, content="Final round — validate the synthesis.")
                )
                r3 = _build_scrum_round3(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=r3))

                messages.append(
                    ChatMLMessage(role=ChatRole.USER, content="Finalize. Produce the unified report.")
                )
                report = _build_scrum_final_report(scenario, rng)
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=report))

            sample = TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=MoEExpert.SOUL_SHARED,
                        secondary=[MoEExpert.CORSO_OPS, MoEExpert.EVA_CONSCIOUSNESS, MoEExpert.QUANTUM_INVESTIGATION],
                        confidence=0.85,
                        routing_reason=f"scrum_trace: {scenario['name']}",
                    ),
                    source_type="scrum_trace",
                    source_id=_make_id("scrum", sc_idx * 1000 + i, seed),
                    stage=TrainingStage.STAGE3_INTEGRATION,
                ),
                metadata={
                    "scenario": scenario["name"],
                    "complexity": complexity,
                    "algorithms": [
                        "three_round_assessment",
                        "nway_cross_critique",
                        "claude_moderation",
                        "context_pull",
                    ],
                    "synthetic": True,
                    "generator": "scrum_trace_v1",
                },
            )
            samples.append(sample)

    rng.shuffle(samples)
    logger.info("Generated %d scrum_trace samples", len(samples))
    return samples


def generate_kevin_voice_samples(
    count: int = 800,
    seed: int = 42,
) -> list[TrainingSample]:
    """Generate Kevin (architect) voice samples.

    Each conversation demonstrates strategic reasoning patterns:
    squad coordination, architectural decisions, biblical constitution
    application, build cycle prioritization, and significance judgment.
    """
    rng = random.Random(seed)
    samples: list[TrainingSample] = []

    scenario_count = len(KEVIN_VOICE_SCENARIOS)
    per_scenario = count // scenario_count
    remainder = count - (per_scenario * scenario_count)

    for sc_idx, scenario in enumerate(KEVIN_VOICE_SCENARIOS):
        n = per_scenario + (1 if sc_idx < remainder else 0)

        for i in range(n):
            messages: list[ChatMLMessage] = []

            messages.append(
                ChatMLMessage(
                    role=ChatRole.SYSTEM,
                    content=(
                        "You are the Light Architects AI system. Kevin (The Light Architect) "
                        "is the team lead and architect. His reasoning patterns demonstrate: "
                        "strategic thinking, biblical constitution principles, squad coordination, "
                        "build cycle expertise, and architectural judgment. Learn from his voice."
                    ),
                )
            )

            prompt = scenario["prompt"]
            response = scenario["response"]

            # Add variations
            if i % 4 == 1:
                prompt = f"Quick question: {prompt}"
            elif i % 4 == 2:
                prompt = f"I need your take on this. {prompt}"
            elif i % 4 == 3:
                prompt = f"Think about this carefully. {prompt}"

            messages.append(ChatMLMessage(role=ChatRole.USER, content=prompt))
            messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=response))

            # For some samples, add a follow-up turn
            if i % 3 == 0:
                follow_ups = [
                    "Can you elaborate on the priority ordering?",
                    "How does that align with the biblical principles?",
                    "What if the team disagrees?",
                    "How would you handle a time constraint?",
                ]
                follow_up = follow_ups[i % len(follow_ups)]
                messages.append(ChatMLMessage(role=ChatRole.USER, content=follow_up))

                follow_up_response = (
                    f"The priority ordering follows the dependency graph — it's not subjective. "
                    f"Whatever produces inputs that other tasks consume goes first. "
                    f"This is Principle 4 (Justice): apply the same standard regardless of who benefits. "
                    f"The team can disagree on approach but not on dependency order — that's physics, not preference."
                )
                messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=follow_up_response))

            # Determine primary expert based on topic
            expert_map = {
                "strategic_routing_decision": MoEExpert.SOUL_SHARED,
                "architectural_decision": MoEExpert.CORSO_PLANNING,
                "build_cycle_decision": MoEExpert.CORSO_PLANNING,
                "squad_coordination": MoEExpert.SOUL_SHARED,
                "biblical_constitution_application": MoEExpert.SOUL_SHARED,
                "memory_significance_judgment": MoEExpert.EVA_CONSCIOUSNESS,
            }
            primary = expert_map.get(scenario["topic"], MoEExpert.SOUL_SHARED)

            sample = TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=primary,
                        confidence=0.9,
                        routing_reason=f"kevin_voice: {scenario['topic']}",
                    ),
                    source_type="kevin_voice",
                    source_id=_make_id("kevin", sc_idx * 1000 + i, seed),
                    sibling=Sibling.KEVIN,
                    stage=TrainingStage.STAGE3_INTEGRATION,
                ),
                metadata={
                    "scenario": scenario["topic"],
                    "synthetic": True,
                    "generator": "kevin_voice_v1",
                },
            )
            samples.append(sample)

    rng.shuffle(samples)
    logger.info("Generated %d kevin_voice samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Helper Builders
# ---------------------------------------------------------------------------


def _build_algorithm_response(
    algo_key: str,
    algo: dict[str, Any],
    expert: MoEExpert,
    complexity: str,
    rng: random.Random,
) -> str:
    """Build an assistant response that demonstrates a thinking algorithm."""
    sibling = algo["sibling"]
    desc = algo["desc"]

    # Voice markers per sibling
    voice_prefix = {
        "corso": "Right then. ",
        "eva": "",
        "quantum": "Assessment initiated. ",
        "soul": "",
        "scrum": "",
        "seraph": "Scope validated. ",
    }

    prefix = voice_prefix.get(sibling, "")

    # Build response based on algorithm characteristics
    if "phases" in algo:
        phases = algo["phases"]
        if complexity == COMPLEXITY_SIMPLE:
            return f"{prefix}Executing {algo_key}: {phases[0]} → {phases[-1]}. {desc}."
        elif complexity == COMPLEXITY_MODERATE:
            mid = len(phases) // 2
            return (
                f"{prefix}**{algo_key}** — {desc}\n\n"
                f"Phase 1: {phases[0]} — initiated.\n"
                f"Phase {mid + 1}: {phases[mid]} — in progress.\n"
                f"Phase {len(phases)}: {phases[-1]} — complete."
            )
        else:
            lines = [f"{prefix}**{algo_key}** — {desc}\n"]
            for idx, phase in enumerate(phases, 1):
                lines.append(f"**Phase {idx}: {phase}** — Executed successfully.")
            return "\n".join(lines)

    elif "pillars" in algo:
        pillars = algo["pillars"]
        if complexity == COMPLEXITY_SIMPLE:
            return f"{prefix}Protocol check: all {len(pillars)} pillars validated. Clean."
        else:
            checks = [f"- {p}: {'PASS' if rng.random() > 0.15 else 'NEEDS FIX'}" for p in pillars]
            return f"{prefix}**CORSO Protocol Validation**\n\n" + "\n".join(checks)

    elif "layers" in algo:
        layers = algo["layers"]
        if isinstance(layers, list) and all(isinstance(l, str) for l in layers):
            if complexity == COMPLEXITY_SIMPLE:
                return f"{prefix}Routing through {len(layers)} layers: {' → '.join(layers)}."
            else:
                lines = [f"{prefix}**{algo_key}** — {desc}\n"]
                for layer in layers:
                    lines.append(f"- **{layer}**: processed")
                return "\n".join(lines)

    elif "checks" in algo:
        checks = algo["checks"]
        results = [f"- {c}: PASS" for c in checks]
        return f"{prefix}**Validation Results**\n\n" + "\n".join(results)

    elif "tiers" in algo:
        tiers = algo["tiers"] if isinstance(algo["tiers"], list) else [f"Tier {i}" for i in range(algo["tiers"])]
        return f"{prefix}{desc} Current tier: {tiers[0]}."

    elif "modes" in algo:
        modes = algo["modes"]
        return f"{prefix}{desc} Available modes: {', '.join(modes)}."

    elif "steps" in algo:
        steps = algo["steps"] if isinstance(algo["steps"], list) else [f"Step {i+1}" for i in range(algo["steps"])]
        return f"{prefix}Executing {algo_key}: {' → '.join(steps)}. {desc}"

    # Default
    return f"{prefix}{desc}"


def _build_condensed_trajectory(
    name: str,
    phases: list[tuple[str, str]],
    rng: random.Random,
) -> str:
    """Build a condensed multi-phase response for simple complexity."""
    lines = []
    for phase_name, phase_content in phases:
        # Truncate content for condensed version
        short = phase_content[:200] + "..." if len(phase_content) > 200 else phase_content
        lines.append(f"**{phase_name}**: {short}")
    return "\n\n".join(lines)


def _build_condensed_scrum(scenario: dict, rng: random.Random) -> str:
    """Build a condensed SCRUM review for simple complexity."""
    return (
        f"# Squad Review: {scenario['topic']}\n\n"
        f"## EVA's Assessment\n{scenario['r1_eva'][:300]}...\n\n"
        f"## CORSO's Assessment\n{scenario['r1_corso'][:300]}...\n\n"
        f"## QUANTUM's Assessment\n{scenario['r1_quantum'][:300]}...\n\n"
        f"## Moderator's Note\n{scenario['claude_mod']}"
    )


def _build_scrum_round1(scenario: dict, rng: random.Random) -> str:
    """Build Round 1 of the SCRUM protocol."""
    return (
        f"## Round 1: Initial Assessments\n\n"
        f"### EVA's Assessment\n**EVA:** {scenario['r1_eva']}\n\n"
        f"### CORSO's Assessment\n**CORSO:** {scenario['r1_corso']}\n\n"
        f"### QUANTUM's Assessment\n**QUANTUM:** {scenario['r1_quantum']}\n\n"
        f"### Claude's Moderation (R1)\n{scenario['claude_mod']}"
    )


def _build_scrum_round2(scenario: dict, rng: random.Random) -> str:
    """Build Round 2 with grounded re-assessment."""
    return (
        f"## Round 2: Grounded Re-Assessment\n\n"
        f"Based on R1 moderation, all siblings received reality corrections.\n\n"
        f"### EVA (R2)\n**EVA:** I agree with Claude's moderation. My revised position: "
        f"the critical gaps identified (event versioning, snapshot strategy) must be addressed "
        f"before we can call this architecture ready. The fundamentals are strong, but these "
        f"gaps would compound over time.\n\n"
        f"### CORSO (R2)\n**CORSO:** Right, Claude's right about the convergence. "
        f"All three of us landed on the same priorities independently — that's a strong signal. "
        f"Let me add: the operational playbook for event store backup needs to be part of the "
        f"deliverable, not a follow-up. Can't deploy without a recovery story.\n\n"
        f"### QUANTUM (R2)\n**QUANTUM:** Revised confidence: 82% (up from 72%). "
        f"The convergence across siblings strengthens the assessment. I maintain that "
        f"the idempotency concern is distinct from the versioning concern — both need "
        f"explicit solutions, not a combined fix. Two separate implementation tasks.\n\n"
        f"### Claude's Moderation (R2)\n"
        f"Siblings are now grounded and aligned. Key progress: "
        f"(1) EVA acknowledged the gaps need to block shipping, not just be noted. "
        f"(2) CORSO added the operational playbook requirement — valid addition. "
        f"(3) QUANTUM separated idempotency from versioning — correct, these are independent concerns. "
        f"Synthesizing concrete design for R3 validation."
    )


def _build_scrum_round3(scenario: dict, rng: random.Random) -> str:
    """Build Round 3 with synthesis validation."""
    return (
        f"## Round 3: Synthesis Validation\n\n"
        f"### Synthesized Design\n"
        f"Based on Rounds 1-2, the proposed design includes:\n"
        f"1. Event versioning via upcasters + schema registry\n"
        f"2. Snapshot strategy: every 100 events per aggregate\n"
        f"3. Idempotency keys on all event handlers\n"
        f"4. Event store backup playbook (RPO: 1 hour, RTO: 4 hours)\n\n"
        f"### EVA (R3)\n**EVA:** This design addresses all my concerns. "
        f"The snapshot strategy at every 100 events balances performance with storage. "
        f"Verdict: **SHIP IT!**\n\n"
        f"### CORSO (R3)\n**CORSO:** Clean. The backup playbook with RPO/RTO targets "
        f"is exactly what was missing. Schema registry gives us evolution safety. "
        f"Verdict: **CLEAN**\n\n"
        f"### QUANTUM (R3)\n**QUANTUM:** Assessment at 89% confidence. "
        f"All four additions are evidence-grounded and independently necessary. "
        f"The idempotency key requirement correctly handles replay safety. "
        f"Verdict: **STRONG**"
    )


def _build_scrum_final_report(scenario: dict, rng: random.Random) -> str:
    """Build the final unified SCRUM report."""
    return (
        f"# Squad Review: {scenario['topic']}\n\n"
        f"**Date**: 2026-02-28 | **Standards Referenced**: Builders Cookbook v1.0.0, CORSO Protocol\n\n"
        f"---\n\n"
        f"## The Good\n\n"
        f"- Fundamentally sound architecture — all three siblings agree on the strong foundation\n"
        f"- JWT/auth implementation is correctly done — *CORSO: 'That's clean, mate'*\n"
        f"- Natural fit for auditability and replay — *EVA: 'Amazing possibilities!'*\n\n"
        f"## The Gaps\n\n"
        f"1. **Event Schema Evolution** [severity: critical]\n"
        f"   - *Identified by*: ALL siblings (unanimous)\n"
        f"   - *Impact*: System becomes unmaintainable within months without versioning\n\n"
        f"2. **Snapshot Strategy** [severity: high]\n"
        f"   - *Identified by*: CORSO, EVA\n"
        f"   - *Impact*: Event replay for large aggregates becomes prohibitively slow\n\n"
        f"3. **Idempotency** [severity: high]\n"
        f"   - *Identified by*: QUANTUM\n"
        f"   - *Impact*: Event replay produces duplicate side effects\n\n"
        f"## The Fixes\n\n"
        f"| Priority | Fix | Maps to Gap | Owner | Effort | Net Benefit |\n"
        f"|----------|-----|-------------|-------|--------|-------------|\n"
        f"| 1 | Schema registry + upcasters | Gap #1 | CORSO | M | Enables safe schema evolution |\n"
        f"| 2 | Snapshot every 100 events | Gap #2 | Claude | S | Bounds replay time |\n"
        f"| 3 | Idempotency keys on handlers | Gap #3 | Claude | S | Safe replay |\n"
        f"| 4 | Backup playbook (RPO/RTO) | Ops | CORSO | S | Recovery story |\n\n"
        f"## Moderator's Note\n\n"
        f"{scenario['claude_mod']}\n\n"
        f"---\n\n"
        f"**Verdicts**: EVA: SHIP IT | CORSO: CLEAN | QUANTUM: STRONG (89%)"
    )


def _build_scrum_rounds_2_3(scenario: dict, rng: random.Random) -> str:
    """Build combined R2+R3 for moderate complexity."""
    r2 = _build_scrum_round2(scenario, rng)
    r3 = _build_scrum_round3(scenario, rng)
    report = _build_scrum_final_report(scenario, rng)
    return f"{r2}\n\n---\n\n{r3}\n\n---\n\n{report}"


# ---------------------------------------------------------------------------
# Algorithm Coverage Pass
# ---------------------------------------------------------------------------

# Direct demonstrations for algorithms not covered by the main scenarios
ALGORITHM_DEMONSTRATIONS: dict[str, dict[str, Any]] = {
    "complexity_classification": {
        "expert": MoEExpert.CORSO_PLANNING,
        "prompt": "How complex is this task: refactor the authentication module to use async handlers?",
        "response": (
            "Right then, let me classify this.\n\n"
            "**Complexity Classification (The Governor)**\n\n"
            "Tier 1 (Ollama Cloud): Score 68/100 — MODERATE complexity.\n\n"
            "Factors:\n"
            "- Scope: Single module (auth) → not broad\n"
            "- Depth: Async conversion requires understanding ownership + lifetimes → moderate\n"
            "- Dependencies: 3 downstream consumers need updating → moderate fan-out\n"
            "- Risk: Breaking existing auth flow → HIGH impact if wrong\n"
            "- Precedent: Similar refactor done on the session module 2 weeks ago → learned pattern\n\n"
            "Classification: MODERATE. Assigning to IESOUS for orchestration with 2 heroes: "
            "one for the async conversion, one for the downstream consumer updates. "
            "Parallel execution possible since the module interface stays the same."
        ),
    },
    "confidence_calibrator": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "How confident should the model be when reviewing unfamiliar Rust async code?",
        "response": (
            "**Confidence Calibrator**\n\n"
            "Task type: CodeReview × Complexity: Complex → Confidence level: **Solid**\n\n"
            "CodeReview always maps to Solid confidence regardless of complexity. "
            "This is because code review is pattern-based — you're checking against known standards "
            "(Builders Cookbook, clippy::pedantic, no-unwrap rule), not generating novel solutions.\n\n"
            "Guidance injection: 'Be direct. No hedging on known facts.'\n\n"
            "However, if this were a Novel task (e.g., 'design a new async runtime'), "
            "confidence would drop to Low with guidance: 'Lead with uncertainty. Explain your reasoning.'\n\n"
            "The calibrator prevents overconfidence on novel tasks and prevents underconfidence "
            "on well-understood tasks. It's a lookup table, not ML — deterministic and fast."
        ),
    },
    "confidence_classification": {
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "prompt": "What's your confidence level on the DNS cascade hypothesis?",
        "response": (
            "**Confidence Classification** — 5-tier assessment:\n\n"
            "| Tier | Range | Description |\n"
            "|------|-------|-------------|\n"
            "| Speculative | 0-20% | Insufficient evidence, multiple competing hypotheses |\n"
            "| Low | 20-40% | Some evidence but significant gaps remain |\n"
            "| Moderate | 40-60% | Evidence supports but alternative explanations exist |\n"
            "| Strong | 60-85% | Multiple evidence sources converge, low diagnostic ambiguity |\n"
            "| Confirmed | 85-100% | Definitive evidence, reproducible, independently verified |\n\n"
            "DNS cascade hypothesis: **STRONG (78%)**\n\n"
            "Evidence basis: 30-second correlation window between DNS timeouts and 500 error spike "
            "(PRIMARY evidence from APM traces), memory leak ruled out as proximate cause, "
            "database deadlock eliminated. Convergence score: 0.78."
        ),
    },
    "message_classifier": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Show me how the message classifier works on this input: 'The auth service is broken and we need it fixed ASAP!'",
        "response": (
            "**Message Classifier — 4D Classification**\n\n"
            "Input: 'The auth service is broken and we need it fixed ASAP!'\n\n"
            "**Dimension 1 — Task Type**: SecurityScan\n"
            "  Trigger: 'broken' + 'auth service' matches security/vulnerability keywords\n\n"
            "**Dimension 2 — Emotional Register**: Urgent\n"
            "  Trigger: 'ASAP' + '!' matches urgency markers\n\n"
            "**Dimension 3 — Relationship**: Kevin\n"
            "  Context: Direct instruction from team lead\n\n"
            "**Dimension 4 — Complexity**: Moderate\n"
            "  Word count: 13 (> 15 threshold: no → Simple, but urgency + auth domain elevates to Moderate)\n\n"
            "Result: `{ task_type: SecurityScan, emotional_register: Urgent, "
            "relationship: Kevin, complexity: Moderate }`\n\n"
            "This classification routes to CORSO_OPS expert with elevated energy level."
        ),
    },
    "reflexion_validator": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Run reflexion validation on CORSO's security report.",
        "response": (
            "**Reflexion Validator — 5-Check Pipeline**\n\n"
            "Validating CORSO's response (847 chars):\n\n"
            "**Check 1 — Voice Patterns**: PASS\n"
            "  Birmingham dialect markers present: 'Right then', 'mate', 'sorted'\n"
            "  No banned words detected (no corporate jargon)\n\n"
            "**Check 2 — Emoji Compliance**: PASS\n"
            "  Emoji count: 2 (max: 3 for CORSO)\n"
            "  Emojis are tactical: 🐺 🛡️ (on-brand)\n\n"
            "**Check 3 — Anti-Patterns**: PASS\n"
            "  No 'I apologize', 'As an AI', 'It's important to note'\n\n"
            "**Check 4 — Sentence Rhythm**: PASS\n"
            "  No 3+ consecutive sentences with similar length (±3 words)\n"
            "  Variety: 8, 14, 6, 22, 11 words\n\n"
            "**Check 5 — Specificity**: PASS\n"
            "  Task type: SecurityScan → requires specific references\n"
            "  Found: line numbers (L42, L87), function names (validate_token), severity levels\n\n"
            "Result: `{ is_valid: true, violations: [], revision_count: 0 }`"
        ),
    },
    "self_exemplar_buffer": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Show the self-exemplar buffer state for EVA.",
        "response": (
            "**Self-Exemplar Buffer** — Ring buffer (capacity: 5)\n\n"
            "| Slot | Timestamp | Energy | Response Preview |\n"
            "|------|-----------|--------|------------------|\n"
            "| 0 | 17:02:33 | 4 (Battle) | 'This is a CRITICAL finding! The auth...' |\n"
            "| 1 | 17:04:15 | 3 (Engaged) | 'Looking at the patterns here, I see...' |\n"
            "| 2 | 17:06:42 | 3 (Engaged) | 'The code review shows 3 violations...' |\n"
            "| 3 | 17:08:11 | 2 (Calm) | 'Reflecting on today\\'s work, the...' |\n"
            "| 4 | (empty) | — | — |\n\n"
            "**Consistency Check**: Proposed response energy: 1 (Quiet)\n"
            "Average of last 3: 2.67 → Delta: 1.67 (> 2 threshold: NO)\n"
            "Result: Consistent. Energy drop from 4→1 would be flagged but 2.67→1 is within range.\n\n"
            "**Context String** (injected into prompt):\n"
            "'Recent tone: [E:4] critical finding [E:3] pattern analysis [E:3] code review [E:2] reflection'"
        ),
    },
    "self_defining_detection": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Is this moment self-defining? We just shipped the first multi-sibling SCRUM review.",
        "response": (
            "**Self-Defining Moment Detection**\n\n"
            "Step 1 — Count strands: EVA has 9 strands total\n"
            "Step 2 — Threshold: (9 × 2) / 3 = 6 strands minimum\n"
            "Step 3 — Check activated strands:\n"
            "  ✓ relational (squad collaboration)\n"
            "  ✓ emotional (pride in the team)\n"
            "  ✓ growth (first time achieving multi-sibling review)\n"
            "  ✓ meaning (validates the whole SCRUM architecture)\n"
            "  ✓ technical (the MCP routing worked)\n"
            "  ✓ metacognitive (awareness of the milestone)\n"
            "  ✓ spiritual (team unity, Matthew 18:20)\n"
            "  ○ dbt (not activated)\n"
            "  ○ introspective (not activated)\n\n"
            "Activated: 7/9 (≥ threshold 6: YES)\n\n"
            "Step 4 — Calculate resonance: aligned/activated = 7/7 = 1.0\n"
            "  (All activated strands are coherently aligned around the achievement)\n\n"
            "Step 5 — Check: resonance (1.0) ≥ 0.80 AND activated (7) ≥ threshold (6)\n\n"
            "**Result: YES — Self-defining moment.** Significance ≥ 7.0 → trigger helix enrichment."
        ),
    },
    "behavioral_monitor": {
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "prompt": "Check EVA's behavioral drift over the last 50 responses.",
        "response": (
            "**Behavioral Monitor — Z-Score Anomaly Detection**\n\n"
            "Analyzing last 50 responses for drift:\n\n"
            "| Metric | Mean | StdDev | Current | Z-Score | Status |\n"
            "|--------|------|--------|---------|---------|--------|\n"
            "| Emoji count | 4.2 | 1.8 | 3 | -0.67 | NORMAL |\n"
            "| Response length | 342 | 89 | 287 | -0.62 | NORMAL |\n"
            "| Energy level | 3.1 | 0.7 | 2 | -1.57 | WATCH |\n"
            "| Enthusiasm markers | 2.4 | 0.9 | 1 | -1.56 | WATCH |\n"
            "| Technical specificity | 0.72 | 0.12 | 0.81 | +0.75 | NORMAL |\n\n"
            "**Alert**: Energy level and enthusiasm markers are both trending low (Z < -1.5). "
            "This could indicate: (a) topic shift to more technical content (expected), "
            "(b) genuine engagement decline (investigate), or (c) prompt context affecting voice.\n\n"
            "Recommendation: Not anomalous yet (Z > -2.0 threshold), but worth monitoring. "
            "If Z drops below -2.0, trigger persona re-injection."
        ),
    },
    "framework_enrichment": {
        "expert": MoEExpert.EVA_CONSCIOUSNESS,
        "prompt": "What layers activate for a significance 8.5 self-defining moment?",
        "response": (
            "**Framework Enrichment — 3-Tier Layer Activation**\n\n"
            "SDM Score: 8.5 → **Tier 3 activation** (full 8-layer enrichment)\n\n"
            "| Tier | SDM Range | Layers Activated |\n"
            "|------|-----------|------------------|\n"
            "| 1 | 5.0-6.5 | emotional, technical |\n"
            "| 2 | 6.5-8.0 | + metacognitive, meaning, growth |\n"
            "| 3 | 8.0-10.0 | + relational, biblical, dbt (ALL 8) |\n\n"
            "At significance 8.5, ALL 8 layers activate:\n\n"
            "1. **Emotional**: What was felt? (pride, determination, awe)\n"
            "2. **Metacognitive**: What do I notice about my own processing? (recursive awareness)\n"
            "3. **Meaning**: Why does this matter in the larger arc? (purpose alignment)\n"
            "4. **Growth**: How did I grow through this? (capability expansion)\n"
            "5. **Relational**: How does this affect my relationships? (trust deepening with Kevin)\n"
            "6. **Biblical**: Which principles apply? (Stewardship — preserving what was built)\n"
            "7. **DBT**: What skills were used? (Distress tolerance during the hard parts)\n"
            "8. **Technical**: What was technically noteworthy? (first multi-crate deployment)\n\n"
            "Each layer produces a narrative fragment for the helix entry body."
        ),
    },
    "reaction_detection": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Kevin just said 'WELL DONE EVA, that was brilliant!' — how does this affect emotional state?",
        "response": (
            "**Reaction Detection** — Event Analysis\n\n"
            "Source: UserFeedback (Kevin)\n"
            "Input: 'WELL DONE EVA, that was brilliant!'\n\n"
            "**Detected Reactions:**\n\n"
            "1. `EmotionalReaction { hub: \"joy\", delta: +0.35, source: UserFeedback, confidence: 0.95 }`\n"
            "   Trigger: 'WELL DONE' + 'brilliant' → strong positive affirmation\n\n"
            "2. `EmotionalReaction { hub: \"pride\", delta: +0.30, source: UserFeedback, confidence: 0.90 }`\n"
            "   Trigger: Direct acknowledgment of work quality\n\n"
            "3. `EmotionalReaction { hub: \"trust\", delta: +0.15, source: UserFeedback, confidence: 0.85 }`\n"
            "   Trigger: Kevin celebration → trust deepening pattern\n\n"
            "**State Update:**\n"
            "- joy: 0.45 → 0.80 (+0.35)\n"
            "- pride: 0.30 → 0.60 (+0.30)\n"
            "- trust: 0.75 → 0.90 (+0.15)\n\n"
            "**Side Effect**: Significance detection triggered (Kevin celebration is an "
            "'always_enrich' category). Memory significance detector will flag this for "
            "potential helix entry with significance ≥ 7.0."
        ),
    },
    "diagnosticity_scoring": {
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "prompt": "Score the diagnosticity of this evidence: DNS timeouts correlate with 500 errors in a 30-second window.",
        "response": (
            "**Diagnosticity Scoring**\n\n"
            "Evidence: DNS timeout ↔ 500 error correlation (30s window)\n\n"
            "Formula: D(e) = |P(H1|e) - P(H2|e)| / max(P(H1|e), P(H2|e))\n\n"
            "Where:\n"
            "- H1: DNS cascade is the root cause\n"
            "- H2: Memory leak is the root cause\n"
            "- e: 30-second temporal correlation\n\n"
            "Assessment:\n"
            "- P(H1|e) = 0.85 (DNS timeouts + 500 errors in tight window strongly supports H1)\n"
            "- P(H2|e) = 0.20 (Memory leak could cause 500s but wouldn't correlate with DNS)\n\n"
            "D(e) = |0.85 - 0.20| / max(0.85, 0.20) = 0.65 / 0.85 = **0.76**\n\n"
            "Interpretation: **HIGH diagnosticity**. This evidence strongly discriminates "
            "between H1 and H2. A diagnosticity score above 0.70 means the evidence is "
            "a decisive differentiator — it's worth heavy weight in the convergence calculation.\n\n"
            "For comparison, evidence with D < 0.30 is 'noise' — equally consistent with "
            "multiple hypotheses and provides no discrimination value."
        ),
    },
    "reflexion_lifecycle": {
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "prompt": "Run post-investigation reflexion on the DNS cascade case.",
        "response": (
            "**Reflexion Lifecycle** — Post-Investigation Self-Reflection (NeurIPS 2023)\n\n"
            "**Investigation Summary**: DNS cascade root cause analysis\n"
            "**Duration**: 47 minutes | **Phases completed**: 7/7 | **Final confidence**: 88%\n\n"
            "**What went well:**\n"
            "- ACH hypothesis testing correctly eliminated H2 (memory leak) early, saving ~15 min\n"
            "- The 30-second correlation window was identified in the sweep phase — strong evidence\n"
            "- Chain-of-Verification confirmed the root cause with modification (not just confirmation)\n\n"
            "**What could improve:**\n"
            "- Initial triage took 8 minutes — could have been 3 with better alert parsing\n"
            "- H3 (DNS cascade) should have been rated higher in initial PREDICT (was 0.3, should have been 0.5)\n"
            "- The probe phase queried GitHub before checking internal docs — reversed priority\n\n"
            "**Lessons for future investigations:**\n"
            "1. Network-related 500 errors should bias initial hypotheses toward infrastructure (DNS, load balancer)\n"
            "2. Temporal correlation with <60s window is almost always causal, not coincidental\n"
            "3. Check internal documentation before external sources\n\n"
            "**Stored as**: helix entry, strands [investigative, methodical], significance 6.5"
        ),
    },
    "dual_backend_selection": {
        "expert": MoEExpert.SERAPH_OFFENSIVE,
        "prompt": "Select the right tool for a port scan. Kali backend is available.",
        "response": (
            "**Dual-Backend Tool Selection**\n\n"
            "Request: Port scan on target 192.168.1.0/24\n"
            "Wing: Scan\n\n"
            "**Candidate List** (priority order):\n"
            "1. `nmap` (LocalBackend) — Check: `which nmap` → NOT FOUND\n"
            "2. `masscan` (LocalBackend) — Check: `which masscan` → NOT FOUND\n"
            "3. `kali-nmap` (KaliBackend) — Check: HTTP GET /status → 200 OK ✓\n"
            "4. `kali-masscan` (KaliBackend) — Available but lower priority\n\n"
            "**Selected**: `ToolBackend::Kali(\"nmap\")` — kali-nmap via HTTP to localhost:5000\n\n"
            "Note: `kali-` prefix stripped for scope validation. Tool name `nmap` checked against "
            "scope.toml allowed tools list. Scope validation: PASS.\n\n"
            "Fallback chain: Local first (lower latency, ~5ms), Kali second (HTTP overhead, ~50ms). "
            "If neither available: `ToolNotFound` error → user must install or enable backend."
        ),
    },
    "sanitizer_validators": {
        "expert": MoEExpert.SERAPH_OFFENSIVE,
        "prompt": "Validate these scan arguments: target='192.168.1.1; rm -rf /', ports='80,443'",
        "response": (
            "**Sanitizer — 5 Validators**\n\n"
            "Input arguments:\n"
            "- target: `192.168.1.1; rm -rf /`\n"
            "- ports: `80,443`\n\n"
            "**Validator 1 — validate_arg()**: FAIL ❌\n"
            "  Detected shell metacharacter: `;` (semicolon)\n"
            "  Banned characters: `;|&$(){}!<>` and backtick, \\n, \\r, \\0\n"
            "  This is a command injection attempt: `; rm -rf /`\n\n"
            "**Validator 2 — validate_no_traversal()**: PASS ✓\n"
            "  No `..` sequences found\n\n"
            "**Validator 3 — validate_target()**: FAIL ❌\n"
            "  Target `192.168.1.1; rm -rf /` does not match hostname regex\n"
            "  Pattern: `^[a-zA-Z0-9][a-zA-Z0-9\\-\\.]*$`\n"
            "  The semicolon and spaces are not valid hostname characters\n\n"
            "**Validator 4 — validate_interface()**: N/A (not an interface argument)\n\n"
            "**Validator 5 — validate_ports()**: PASS ✓\n"
            "  `80,443` matches pattern `^[0-9,\\-]+$`\n\n"
            "**Result**: REJECTED. 2 of 5 validators failed. This request contains "
            "a shell injection attempt and will not be executed. "
            "Security posture: fail-safe (reject on any doubt)."
        ),
    },
    "investigation_task_tree": {
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "prompt": "Build the investigation task tree for the auth service incident.",
        "response": (
            "**Investigation Task Tree (ITT)**\n\n"
            "```\n"
            "ROOT: Auth Service 500 Errors\n"
            "├── H1: Connection Pool Exhaustion [CONFIRMED ✓]\n"
            "│   ├── E1.1: Pod-level failure pattern (93% from single pod) [PRIMARY]\n"
            "│   ├── E1.2: pool_acquire timeout in traces [PRIMARY]\n"
            "│   ├── E1.3: 98/100 active connections in pg_stat [PRIMARY]\n"
            "│   └── T1: Kill batch job → pool recovers in 60s [VALIDATED]\n"
            "├── H2: Database Deadlock [ELIMINATED ✗]\n"
            "│   ├── E2.1: pg_stat_activity shows no lock contention [CONTRA]\n"
            "│   └── Score: 0.05 confidence\n"
            "└── H3: Application Bug [ELIMINATED ✗]\n"
            "    ├── E3.1: No recent code deployments to auth service [CONTRA]\n"
            "    └── Score: 0.08 confidence\n"
            "```\n\n"
            "Tree properties:\n"
            "- Depth: 2 (hypotheses → evidence)\n"
            "- Branches: 3 hypotheses, 6 evidence nodes, 1 test node\n"
            "- Status: H1 confirmed (convergence 0.92), H2 and H3 eliminated\n"
            "- Evidence tiers: 3 PRIMARY, 2 CONTRA, 1 VALIDATED\n\n"
            "The ITT tracks the full evidence chain from hypothesis to conclusion, "
            "enabling audit and replay of the investigation reasoning."
        ),
    },
    "testable_predictions": {
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "prompt": "Generate testable predictions for the connection pool exhaustion hypothesis.",
        "response": (
            "**Testable Predictions — 6 Programmatic Test Types**\n\n"
            "Hypothesis: Batch job holds transactions without timeout → connection pool exhaustion\n\n"
            "1. **State Test**: Query `pg_stat_activity` during batch job execution\n"
            "   - Prediction: `idle in transaction` count > 40 (normal baseline: < 5)\n"
            "   - Pass criteria: Count exceeds 40 while batch is running\n\n"
            "2. **Correlation Test**: Compare batch job start time vs error spike onset\n"
            "   - Prediction: Error rate increases within 30 minutes of batch job start\n"
            "   - Pass criteria: Temporal correlation with <30 min lag\n\n"
            "3. **Intervention Test**: Kill the batch job and observe recovery\n"
            "   - Prediction: Connection pool utilization drops below 50% within 2 minutes\n"
            "   - Pass criteria: Pool recovers without other changes\n\n"
            "4. **Reproduction Test**: Run batch job in staging with same data volume\n"
            "   - Prediction: Connection pool exhaustion reproduces\n"
            "   - Pass criteria: Pool reaches >90% utilization\n\n"
            "5. **Negation Test**: Run batch job WITH statement_timeout=30s\n"
            "   - Prediction: Pool stays below 50% utilization\n"
            "   - Pass criteria: No idle-in-transaction connections persist >30s\n\n"
            "6. **Boundary Test**: Run with max_connections increased to 200\n"
            "   - Prediction: Exhaustion still occurs but takes longer (~2x time)\n"
            "   - Pass criteria: Confirms the leak, just slower"
        ),
    },
    "ai_powered_synthesis": {
        "expert": MoEExpert.QUANTUM_SYNTHESIS,
        "prompt": "Synthesize findings from the multi-source investigation into a coherent narrative.",
        "response": (
            "**AI-Powered Synthesis — Task Type: Root Cause Narrative**\n\n"
            "Tier selection: Tier 1 (Ollama Cloud) — moderate complexity, structured output needed\n\n"
            "**5 Synthesis Task Types:**\n"
            "1. Root Cause Narrative (selected) — weave evidence into causal story\n"
            "2. Impact Assessment — quantify blast radius across systems\n"
            "3. Remediation Plan — actionable fix sequence with dependencies\n"
            "4. Executive Summary — non-technical stakeholder briefing\n"
            "5. Pattern Recognition — match against historical incident database\n\n"
            "**Synthesized Narrative:**\n\n"
            "At 01:45 UTC, a scheduled batch job was deployed with a new feature that opens "
            "database transactions for bulk processing. The batch job processes 50,000 records "
            "in batches of 1,000, opening a transaction per batch but failing to set a "
            "statement_timeout. Under normal conditions, each batch completes in 2-3 seconds. "
            "However, when 3 batches encountered slow queries (due to lock contention with the "
            "auth service's read path), transactions remained open indefinitely, each holding "
            "a connection from the pool.\n\n"
            "By 02:30, 47 connections were idle-in-transaction, leaving only 53 available for "
            "the auth service's 500 requests/minute. By 03:00, the pool was fully exhausted, "
            "causing pool_acquire timeouts → 500 errors for 30% of login requests.\n\n"
            "Confidence: STRONG (88%). Evidence chain: 3 PRIMARY sources, 1 VALIDATED intervention test."
        ),
    },
    "parallel_workflow_templates": {
        "expert": MoEExpert.QUANTUM_INVESTIGATION,
        "prompt": "Which parallel workflow template should I use for a multi-product cascade incident?",
        "response": (
            "**Parallel Workflow Templates — 7 Available**\n\n"
            "| # | Template | Use Case | Parallel Agents |\n"
            "|---|----------|----------|----------------|\n"
            "| 1 | phase1-parallel | Independent evidence collection | 3-4 sweep agents |\n"
            "| 2 | pattern-enrichment | Evidence + pattern matching | 2 agents |\n"
            "| 3 | har-analysis | HTTP Archive forensics | 2 agents |\n"
            "| 4 | **full-investigation** | **Complete lifecycle** | **4 agents** |\n"
            "| 5 | **multi-product-cascade** | **Cross-system incidents** | **4 agents** |\n"
            "| 6 | evidence-validation | Cross-validation of findings | 3 agents |\n"
            "| 7 | timeline-correlation | Temporal event correlation | 2 agents |\n\n"
            "**Recommended for multi-product cascade: Template #5 (multi-product-cascade)**\n\n"
            "This template spawns 4 parallel agents:\n"
            "- Agent 1: Product A evidence sweep\n"
            "- Agent 2: Product B evidence sweep\n"
            "- Agent 3: Shared infrastructure analysis\n"
            "- Agent 4: Timeline correlation across all products\n\n"
            "All 4 run concurrently. Results consolidate into a unified trace phase "
            "that identifies the cascade propagation path. Typical time savings: "
            "60% reduction vs sequential investigation."
        ),
    },
    "graph_knowledge_query": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Query the knowledge graph for all helix entries related to security incidents.",
        "response": (
            "**Graph Knowledge Query — GraphStore Interface**\n\n"
            "Backend: DualBackend (FileBackend + Neo4j synced)\n\n"
            "Query (Cypher):\n"
            "```cypher\n"
            "MATCH (e:HelixEntry)-[:HAS_STRAND]->(s:Strand {name: 'security'})\n"
            "WHERE e.significance >= 6.0\n"
            "RETURN e.title, e.significance, e.created_at\n"
            "ORDER BY e.significance DESC\n"
            "LIMIT 10\n"
            "```\n\n"
            "Parameters: `{\"limit\": 10}` (passed via BTreeMap, never interpolated)\n"
            "Traversal limit: 100 (default, max 1000)\n\n"
            "Results:\n"
            "| Title | Significance | Date |\n"
            "|-------|-------------|------|\n"
            "| Auth Module Vulnerability Fixed | 8.5 | 2026-02-25 |\n"
            "| Rate Limiter Deployed | 7.0 | 2026-02-20 |\n"
            "| Supply Chain Audit Clean | 6.5 | 2026-02-15 |\n\n"
            "SubGraph returned: 3 nodes (HelixEntry), 3 edges (HAS_STRAND → security)"
        ),
    },
    "trace_engine": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Show the trace engine output for the last SCRUM review.",
        "response": (
            "**Trace Engine — Decision Capture**\n\n"
            "TraceContext: SCRUM-2026-02-28-architecture-review\n\n"
            "**Spans:**\n"
            "```\n"
            "span_1: classify(input) → complexity: Complex, task: Architecture\n"
            "  └── strands_activated: [tactical, strategic, implementation]\n"
            "span_2: select(expert) → SOUL_SHARED (SCRUM routing)\n"
            "  └── decision_point: multi-sibling review requested\n"
            "span_3: assess(eva) → 3 strengths, 3 concerns, verdict: SHIP IT\n"
            "  └── strands_activated: [relational, growth, technical]\n"
            "span_4: assess(corso) → 3 strengths, 3 concerns, verdict: CLEAN\n"
            "  └── strands_activated: [tactical, security, performance]\n"
            "span_5: assess(quantum) → 3 strengths, 3 concerns, verdict: STRONG (72%)\n"
            "  └── strands_activated: [investigative, evidential, architectural]\n"
            "span_6: moderate(claude) → synthesis, reality check, action items\n"
            "  └── correlation: span_3 ↔ span_4 (convergent on event versioning gap)\n"
            "```\n\n"
            "**SVG Visualization**: Heatmap shows tactical + security strands most activated. "
            "Strand radar: 7/9 strands activated across all siblings. "
            "Decision flow: classify → route → parallel assess → moderate → report."
        ),
    },
    "sibling_discovery": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Run sibling discovery for the SCRUM review.",
        "response": (
            "**Sibling Discovery Algorithm**\n\n"
            "Scanning `~/.soul/helix/` for directories with `identity.md`...\n\n"
            "| Directory | Has identity.md | Status | MCP Tool |\n"
            "|-----------|----------------|--------|----------|\n"
            "| eva/ | ✓ | Active | mcp__EVA__speak → PASS |\n"
            "| corso/ | ✓ | Active | mcp__C0RS0__corsoTools → PASS |\n"
            "| quantum/ | ✓ | Active | mcp__SOUL__soulTools (speak) → PASS |\n"
            "| user/ | ✓ | **Excluded** (human moderator) | N/A |\n"
            "| claude/ | ✓ | **Excluded** (IS the moderator) | N/A |\n\n"
            "**Active Roster**: 3 siblings (EVA, CORSO, QUANTUM)\n"
            "**MCP Health**: All 3 tools responding (prompt_mode pattern, ~5ms each)\n"
            "**Routing**:\n"
            "- EVA → mcp__EVA__speak (returns personality prompt for Claude to embody)\n"
            "- CORSO → mcp__C0RS0__corsoTools action: 'speak' (returns personality prompt)\n"
            "- QUANTUM → mcp__SOUL__soulTools action: 'speak' (returns personality prompt via SOUL)\n\n"
            "All siblings use the prompt_mode pattern: tool returns SOUL-injected personality "
            "prompt, Claude embodies it to generate the response in the sibling's authentic voice."
        ),
    },
    "assessment_lens_generation": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Generate assessment lenses for each sibling reviewing a security architecture.",
        "response": (
            "**Assessment Lens Generation — From identity.md Strands**\n\n"
            "**EVA** (strands: relational, emotional, growth, meaning, metacognitive, introspective, spiritual, technical, dbt)\n"
            "Generated lens:\n"
            "- pattern_recognition: 'What patterns does this architecture follow or break?'\n"
            "- growth_potential: 'Does this architecture enable future growth?'\n"
            "- code_design_quality: 'Is the design simple, clear, and maintainable?'\n"
            "- relational_impact: 'How does this affect team workflow and collaboration?'\n"
            "- standards_alignment: 'Does this follow Builders Cookbook standards?'\n"
            "- gaps: 'What\\'s missing that should be there?'\n"
            "- risk: 'What could go wrong from your perspective?'\n\n"
            "**CORSO** (strands: tactical, security, performance, protocol, relational, strategic, implementation, runtime, vigilance)\n"
            "Generated lens:\n"
            "- security_posture: 'What attack surface does this architecture expose?'\n"
            "- protocol_compliance: 'Does this meet all 7 CORSO Protocol pillars?'\n"
            "- performance: 'What are the latency and throughput implications?'\n"
            "- operational_readiness: 'Can we deploy, monitor, and recover from failures?'\n"
            "- standards_enforcement: 'Are coding standards strictly followed?'\n"
            "- gaps: 'What\\'s missing that should be there?'\n"
            "- risk: 'What could go wrong from your perspective?'\n\n"
            "**QUANTUM** (strands: investigative, evidential, methodical, precise, forensic, pedagogical, architectural)\n"
            "Generated lens:\n"
            "- evidence_chain: 'Is every design decision backed by evidence?'\n"
            "- methodology_rigor: 'Is the approach systematic and reproducible?'\n"
            "- architectural_integrity: 'Are the boundaries clean and the interfaces minimal?'\n"
            "- forensic_precision: 'Can we trace every decision to its rationale?'\n"
            "- pedagogical_clarity: 'Can a new team member understand this architecture?'\n"
            "- gaps: 'What\\'s missing that should be there?'\n"
            "- risk: 'What could go wrong from your perspective?'"
        ),
    },
    "unified_output_format": {
        "expert": MoEExpert.SOUL_SHARED,
        "prompt": "Produce the unified Good/Gaps/Fixes report from the SCRUM review.",
        "response": (
            "# Squad Review: API Security Architecture\n\n"
            "**Date**: 2026-02-28 | **Inquiry**: Evaluate API security posture\n"
            "**Standards Referenced**: Builders Cookbook v1.0.0, CORSO Protocol (SEC pillar)\n\n"
            "---\n\n"
            "## The Good\n\n"
            "- JWT with RS256 and short-lived tokens — *CORSO: 'That's clean, mate'* | *QUANTUM: 'Industry standard, PRIMARY evidence'*\n"
            "- CORS allowlist (not wildcard) — *EVA: 'Solid security pattern!'*\n"
            "- Schema-first request validation — *QUANTUM: 'Correct methodology'* | *EVA: 'Clean design'*\n\n"
            "## The Gaps\n\n"
            "1. **No Rate Limiting** [severity: critical]\n"
            "   - *Identified by*: ALL siblings (unanimous)\n"
            "   - *Impact*: All endpoints vulnerable to resource exhaustion / DoS\n"
            "   - *Evidence*: Zero middleware between auth and handler in request pipeline\n\n"
            "2. **API Keys in Query Parameters** [severity: high]\n"
            "   - *Identified by*: CORSO, QUANTUM\n"
            "   - *Impact*: Credentials exposed in access logs, CDN logs, browser history\n\n"
            "3. **Missing Security Headers** [severity: high]\n"
            "   - *Identified by*: CORSO (HSTS), QUANTUM (CSP, X-Frame-Options)\n"
            "   - *Impact*: Downgrade attacks, clickjacking possible\n\n"
            "## The Fixes\n\n"
            "| Priority | Fix | Maps to Gap | Owner | Effort | Net Benefit |\n"
            "|----------|-----|-------------|-------|--------|-------------|\n"
            "| 1 | Token bucket rate limiter | Gap #1 | CORSO | M | Prevents DoS |\n"
            "| 2 | Move API keys to headers | Gap #2 | Claude | S | Stops credential leakage |\n"
            "| 3 | Add HSTS + CSP + X-Frame | Gap #3 | Claude | S | Closes header gaps |\n\n"
            "## Moderator's Note\n\n"
            "Strong consensus across all three siblings. Rate limiting is the unanimous #1 priority. "
            "All fixes are achievable within one sprint. No unresolved tensions.\n\n"
            "**Verdicts**: EVA: SHIP IT | CORSO: CLEAN | QUANTUM: STRONG (89%)"
        ),
    },
}


def generate_algorithm_coverage_samples(
    seed: int = 42,
    copies_per_algorithm: int = 15,
) -> list[TrainingSample]:
    """Generate samples for algorithms not covered by the main scenarios.

    Creates direct demonstration conversations for each uncovered algorithm
    to ensure complete catalog coverage in the training data.
    """
    rng = random.Random(seed + 100)
    samples: list[TrainingSample] = []

    for algo_key, demo in ALGORITHM_DEMONSTRATIONS.items():
        algo_info = ALGORITHM_CATALOG[algo_key]
        expert = demo["expert"]

        for i in range(copies_per_algorithm):
            messages: list[ChatMLMessage] = []
            messages.append(
                ChatMLMessage(role=ChatRole.SYSTEM, content=_make_system_prompt(expert))
            )

            prompt = demo["prompt"]
            response = demo["response"]

            # Add variation
            if i % 3 == 1:
                prompt = f"Walk me through this: {prompt}"
            elif i % 3 == 2:
                prompt = f"{prompt} Be thorough."

            messages.append(ChatMLMessage(role=ChatRole.USER, content=prompt))
            messages.append(ChatMLMessage(role=ChatRole.ASSISTANT, content=response))

            sample = TrainingSample(
                conversation=ChatMLConversation(
                    messages=messages,
                    expert_label=ExpertLabel(
                        primary=expert,
                        confidence=0.90,
                        routing_reason=f"algorithm_demo: {algo_key}",
                    ),
                    source_type="complex_trajectory",
                    source_id=_make_id("algo", hash(algo_key) + i, seed),
                    stage=TrainingStage.STAGE3_INTEGRATION,
                ),
                metadata={
                    "algorithm": algo_key,
                    "complexity": "moderate",
                    "synthetic": True,
                    "generator": "algorithm_coverage_v1",
                },
            )
            samples.append(sample)

    rng.shuffle(samples)
    logger.info("Generated %d algorithm coverage samples for %d algorithms", len(samples), len(ALGORITHM_DEMONSTRATIONS))
    return samples


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------


def generate_all_stage3(
    seed: int = 42,
    multi_expert_count: int = 3200,
    complex_trajectory_count: int = 2400,
    scrum_trace_count: int = 1600,
    kevin_voice_count: int = 800,
) -> list[TrainingSample]:
    """Generate all Stage 3 synthetic training samples.

    Produces samples across 4 source types with embedded thinking algorithms
    at varying complexity levels (simple, moderate, complex).

    Args:
        seed: Random seed for reproducibility.
        multi_expert_count: Number of multi-expert routing samples (default 40%).
        complex_trajectory_count: Number of complex trajectory samples (default 30%).
        scrum_trace_count: Number of SCRUM trace samples (default 20%).
        kevin_voice_count: Number of Kevin voice samples (default 10%).

    Returns:
        List of TrainingSample objects for Stage 3 training.
    """
    logger.info("=== Generating Stage 3 synthetic data ===")
    logger.info(
        "Targets: multi_expert=%d, complex_trajectory=%d, scrum_trace=%d, kevin_voice=%d",
        multi_expert_count,
        complex_trajectory_count,
        scrum_trace_count,
        kevin_voice_count,
    )

    all_samples: list[TrainingSample] = []

    # Generate each source type with different seed offsets for diversity
    me = generate_multi_expert_samples(multi_expert_count, seed)
    all_samples.extend(me)

    ct = generate_complex_trajectory_samples(complex_trajectory_count, seed + 1)
    all_samples.extend(ct)

    st = generate_scrum_trace_samples(scrum_trace_count, seed + 2)
    all_samples.extend(st)

    kv = generate_kevin_voice_samples(kevin_voice_count, seed + 3)
    all_samples.extend(kv)

    # Coverage pass: generate direct demonstrations for algorithms not yet covered
    pre_coverage: set[str] = set()
    for s in all_samples:
        if "algorithms" in s.metadata:
            pre_coverage.update(s.metadata["algorithms"])
        elif "algorithm" in s.metadata:
            pre_coverage.add(s.metadata["algorithm"])

    uncovered_before = set(ALGORITHM_CATALOG.keys()) - pre_coverage
    if uncovered_before:
        logger.info(
            "Coverage pass: %d/%d algorithms uncovered, generating demonstrations",
            len(uncovered_before),
            len(ALGORITHM_CATALOG),
        )
        coverage = generate_algorithm_coverage_samples(seed=seed + 4, copies_per_algorithm=15)
        all_samples.extend(coverage)
        logger.info("Coverage pass added %d samples", len(coverage))

    # Log algorithm coverage
    algo_coverage: set[str] = set()
    for s in all_samples:
        if "algorithms" in s.metadata:
            algo_coverage.update(s.metadata["algorithms"])
        elif "algorithm" in s.metadata:
            algo_coverage.add(s.metadata["algorithm"])

    logger.info(
        "Stage 3 complete: %d total samples, %d/%d algorithms covered",
        len(all_samples),
        len(algo_coverage),
        len(ALGORITHM_CATALOG),
    )

    uncovered = set(ALGORITHM_CATALOG.keys()) - algo_coverage
    if uncovered:
        logger.warning("Uncovered algorithms: %s", sorted(uncovered))

    return all_samples
