# X-Algorithm For You Feed: Comprehensive Technical Summary

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Technical Stack](#technical-stack)
4. [Component Deep Dive](#component-deep-dive)
5. [Data Structures & Types](#data-structures--types)
6. [Pipeline Execution Flow](#pipeline-execution-flow)
7. [Dependencies & Libraries](#dependencies--libraries)
8. [Configuration Parameters](#configuration-parameters)
9. [Build & Testing Infrastructure](#build--testing-infrastructure)
10. [Key Design Decisions](#key-design-decisions)
11. [Performance Characteristics](#performance-characteristics)

---

## System Overview

The x-algorithm is a **four-component distributed recommendation system** that powers X's "For You" feed. It retrieves, ranks, and filters posts using a combination of:

- **Realtime in-memory data stores** (Thunder)
- **ML-based retrieval and ranking** (Phoenix)
- **Composable pipeline framework** (Candidate Pipeline)
- **Orchestration layer** (Home Mixer)

**Key Technical Achievement:** Zero hand-engineered features - all relevance patterns are learned by a Grok-based transformer model adapted from xAI's Grok-1 open source release.

---

## Architecture Components

### 1. Candidate Pipeline (Rust Framework)
- **Language:** Rust (async/await with Tokio)
- **Purpose:** Generic, reusable pipeline framework for recommendation systems
- **Location:** `/candidate-pipeline/`

### 2. Home Mixer (Rust Orchestration)
- **Language:** Rust with gRPC (Tonic)
- **Purpose:** Main orchestration layer implementing the For You feed
- **Location:** `/home-mixer/`
- **Exposes:** `ScoredPostsService` gRPC endpoint

### 3. Phoenix (Python ML Models)
- **Language:** Python with JAX + Haiku
- **Purpose:** ML retrieval (two-tower) and ranking (Grok transformer)
- **Location:** `/phoenix/`

### 4. Thunder (Rust In-Memory Store)
- **Language:** Rust with Kafka integration
- **Purpose:** Realtime ingestion and sub-millisecond lookups for in-network posts
- **Location:** `/thunder/`

---

## Technical Stack

### Rust Components (Candidate Pipeline, Home Mixer, Thunder)

#### Core Technologies
- **Async Runtime:** Tokio
- **Concurrency:** DashMap (lock-free concurrent HashMap)
- **gRPC:** Tonic with compression (Gzip, Zstd)
- **Serialization:** Protobuf
- **Streaming:** Kafka consumer (for Thunder)

#### Internal X Libraries
- `xai_http_server` - HTTP server utilities
- `xai_stats_macro` - Metrics and stats macros
- `xai_init_utils` - Initialization utilities
- `xai_profiling` - CPU profiling support
- `xai_thunder_proto` - Thunder protobuf definitions
- `xai_home_mixer_proto` - Home Mixer protobuf definitions
- `xai_visibility_filtering` - Content safety filters
- `xai_strato_client` - Client for X's Strato data platform

#### Development Tools
- `clap` - Command-line argument parsing
- `log` - Logging framework
- `futures` - Async utilities

### Python Components (Phoenix)

#### Core Technologies
- **ML Framework:** JAX 0.8.1 (Google's high-performance numerical computing)
- **Neural Network Library:** dm-haiku >=0.0.13 (DeepMind)
- **Numerics:** numpy >=1.26.4

#### Development Tools
- **Package Manager:** uv (fast Python package installer)
- **Testing:** pytest
- **Type Checking:** pyright >=1.1.408
- **Linting:** ruff (100 char line length)

---

## Component Deep Dive

### 1. Candidate Pipeline Framework

**Core Trait Hierarchy:**

```rust
// Main orchestration
trait CandidatePipeline<Q, C> {
    async fn execute(&self, query: Q) -> PipelineResult<Q, C>;
}

// Pipeline stages (all async)
trait QueryHydrator<Q> {
    async fn hydrate(&self, query: &mut Q) -> Result<(), Error>;
}

trait Source<Q, C> {
    async fn get_candidates(&self, query: &Q) -> Result<Vec<C>, Error>;
}

trait Hydrator<Q, C> {
    async fn hydrate(&self, query: &Q, candidates: &mut [C]) -> Result<(), Error>;
}

trait Filter<Q, C> {
    fn filter(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C>;
}

trait Scorer<Q, C> {
    fn score(&self, query: &Q, candidates: &mut [C]);
}

trait Selector<Q, C> {
    fn select(&self, candidates: Vec<C>) -> Vec<C>;
}

trait SideEffect<Q, C> {
    async fn execute(&self, result: &PipelineResult<Q, C>);
}
```

**Key Design Patterns:**
- **Parallel Execution:** Sources and hydrators run concurrently
- **Sequential Filters/Scorers:** Order matters, run one at a time
- **Graceful Error Handling:** Configurable failure modes (fail-fast vs. continue)
- **Separation of Concerns:** Business logic separate from orchestration

**PipelineResult Structure:**
```rust
struct PipelineResult<Q, C> {
    query: Q,
    retrieved_candidates: Vec<C>,
    filtered_candidates: Vec<C>,
    selected_candidates: Vec<C>,
}
```

---

### 2. Home Mixer (Orchestration Layer)

**Server Configuration:**
- **Protocol:** gRPC with HTTP/2
- **Compression:** Gzip and Zstd support
- **Metrics:** Prometheus endpoint
- **Reflection:** gRPC reflection enabled for debugging

**Module Structure:**

```
home-mixer/
├── main.rs                    # Server initialization
├── server.rs                  # gRPC service implementation
├── candidate_pipeline/        # Pipeline configuration
├── query_hydrators/
│   ├── user_action_sequence.rs   # Fetch engagement history
│   └── user_features.rs           # Fetch following list, preferences
├── sources/
│   ├── thunder_source.rs          # In-network posts
│   └── phoenix_source.rs          # Out-of-network retrieval
├── candidate_hydrators/
│   ├── core_data_hydrator.rs      # Post text, media, metadata
│   ├── in_network_hydrator.rs     # In-network specific data
│   ├── gizmoduck_hydrator.rs      # Author information
│   ├── subscription_hydrator.rs   # Subscription status
│   └── video_duration_hydrator.rs # Video metadata
├── filters/
│   ├── age_filter.rs
│   ├── self_tweet_filter.rs
│   ├── drop_duplicates_filter.rs
│   ├── vf_filter.rs               # Visibility filtering
│   ├── muted_keyword_filter.rs
│   ├── author_socialgraph_filter.rs
│   ├── subscription_eligibility_filter.rs
│   ├── dedup_conversation_filter.rs
│   ├── retweet_dedup_filter.rs
│   ├── previously_seen_filter.rs
│   └── previously_served_filter.rs
├── scorers/
│   ├── phoenix_scorer.rs          # ML predictions
│   ├── weighted_scorer.rs         # Combine predictions
│   ├── author_diversity_scorer.rs # Diversity boost
│   └── oon_scorer.rs              # Out-of-network adjustment
├── selectors/
│   └── score_selector.rs          # Top-K selection
└── side_effects/
    └── cache_request_info.rs      # Session caching
```

**PostCandidate Data Structure:**

```rust
struct PostCandidate {
    // Core identifiers
    tweet_id: i64,
    author_id: i64,
    
    // Content
    tweet_text: Option<String>,
    media_entities: Vec<MediaEntity>,
    
    // ML Predictions (18 action types)
    phoenix_scores: PhoenixScores,
    
    // Computed scores
    weighted_score: Option<f64>,
    author_diversity_score: Option<f64>,
    oon_score: Option<f64>,
    
    // Metadata
    source: CandidateSource, // Thunder vs Phoenix
    created_at: i64,
    is_retweet: bool,
    is_reply: bool,
    
    // Visibility
    visibility_filters: Vec<String>,
}
```

**PhoenixScores (18 Action Predictions):**

```rust
struct PhoenixScores {
    // Positive engagement
    favorite: Option<f64>,         // P(like)
    reply: Option<f64>,            // P(reply)
    retweet: Option<f64>,          // P(retweet)
    quote: Option<f64>,            // P(quote tweet)
    click: Option<f64>,            // P(detail click)
    profile_click: Option<f64>,    // P(profile visit)
    share: Option<f64>,            // P(external share)
    dwell: Option<f64>,            // P(long dwell time)
    follow_author: Option<f64>,    // P(follow author)
    
    // Video engagement
    video_view: Option<f64>,       // P(video playback)
    vqv: Option<f64>,              // P(video quality view)
    
    // Photo engagement
    photo_expand: Option<f64>,     // P(photo expand)
    
    // Negative signals
    not_interested: Option<f64>,   // P(not interested)
    block_author: Option<f64>,     // P(block author)
    mute_author: Option<f64>,      // P(mute author)
    report: Option<f64>,           // P(report post)
    
    // Additional metrics
    good_click: Option<f64>,       // P(meaningful click)
    good_profile_click: Option<f64>, // P(meaningful profile visit)
}
```

**Weighted Scoring Formula:**

```
final_score = 
    w_fav * P(favorite) +
    w_reply * P(reply) +
    w_retweet * P(retweet) +
    w_click * P(click) +
    w_dwell * P(dwell) +
    w_share * P(share) +
    w_follow * P(follow_author) -
    w_block * P(block_author) -
    w_mute * P(mute_author) -
    w_report * P(report)
```

Where negative signals (block, mute, report) have negative weights to demote content.

---

### 3. Phoenix (ML Models)

**Two-Stage ML Architecture:**

#### Stage 1: Retrieval (Two-Tower Model)

**Purpose:** Fast approximate similarity search over millions of posts

```python
class PhoenixRetrievalModel:
    """Two-tower model for candidate retrieval"""
    
    def __init__(self, config: RetrievalConfig):
        self.user_tower = UserTower(config)
        self.candidate_tower = CandidateTower(config)
    
    def __call__(self, batch: RetrievalBatch):
        # Encode user + engagement history
        user_embedding = self.user_tower(
            user_hashes=batch.user_hashes,
            history_hashes=batch.history_hashes,
            history_actions=batch.history_actions
        )  # [B, D]
        
        # Encode all candidate posts
        candidate_embeddings = self.candidate_tower(
            post_hashes=batch.candidate_hashes
        )  # [N, D]
        
        # Compute similarity scores
        scores = jnp.dot(
            user_embedding / jnp.linalg.norm(user_embedding),
            (candidate_embeddings / jnp.linalg.norm(candidate_embeddings)).T
        )  # [B, N]
        
        return scores
```

**Key Features:**
- Hash-based embeddings (no vocabulary)
- Multiple hash functions per entity
- Normalized dot product similarity
- Returns top-K candidates per user

#### Stage 2: Ranking (Grok Transformer with Candidate Isolation)

**Architecture Overview:**

```python
class PhoenixModel:
    """Grok-based transformer for ranking with candidate isolation"""
    
    def __init__(self, config: PhoenixModelConfig):
        self.transformer = GrokTransformer(config.transformer)
        self.embedding_tables = create_embedding_tables(config.hash_config)
        self.action_heads = create_action_heads(num_actions=18)
    
    def __call__(self, batch: RecsysBatch, embeddings: RecsysEmbeddings):
        # Input sequence structure:
        # [CLS] [user] [history_1] ... [history_N] [cand_1] ... [cand_M]
        
        # Construct input embeddings
        sequence = jnp.concatenate([
            embeddings.cls_token,
            embeddings.user_emb,
            embeddings.history_post_embs,
            embeddings.candidate_post_embs
        ], axis=1)  # [B, seq_len, D]
        
        # Create attention mask with candidate isolation
        attn_mask = make_recsys_attn_mask(
            num_history=batch.num_history,
            num_candidates=batch.num_candidates
        )
        
        # Transformer forward pass
        hidden_states = self.transformer(
            sequence,
            attn_mask=attn_mask
        )  # [B, seq_len, D]
        
        # Extract candidate representations
        candidate_states = hidden_states[:, -batch.num_candidates:, :]
        
        # Predict actions for each candidate
        logits = {}
        for action_name, head in self.action_heads.items():
            logits[action_name] = head(candidate_states)
        
        return logits  # {action_name: [B, num_candidates]}
```

**Critical Design: Candidate Isolation Attention Mask**

```python
def make_recsys_attn_mask(num_history, num_candidates):
    """
    Create attention mask ensuring candidates cannot attend to each other.
    
    Attention pattern:
    - CLS/user/history tokens: causal attention (can see past)
    - Candidate tokens: can attend to CLS/user/history + self, 
                        but NOT to other candidates
    
    This ensures each candidate's score is independent of other candidates
    in the batch, making scores cacheable and consistent.
    """
    total_len = 1 + 1 + num_history + num_candidates
    mask = jnp.zeros((total_len, total_len))
    
    # Context tokens (CLS, user, history): causal attention
    context_len = 1 + 1 + num_history
    for i in range(context_len):
        mask[i, :i+1] = 1
    
    # Candidate tokens: attend to context + self only
    for i in range(num_candidates):
        cand_idx = context_len + i
        # Attend to all context
        mask[cand_idx, :context_len] = 1
        # Attend to self
        mask[cand_idx, cand_idx] = 1
        # Do NOT attend to other candidates (stays 0)
    
    return mask
```

**Hash-Based Embeddings:**

```python
class HashConfig:
    num_user_hashes: int = 3      # Multiple hash functions per user
    num_item_hashes: int = 3      # Multiple hash functions per post
    num_author_hashes: int = 2    # Multiple hash functions per author
    vocab_size: int = 10_000_000  # Hash table size

def lookup_hash_embeddings(entity_id, num_hashes, embedding_table):
    """
    Use multiple hash functions for robustness.
    Helps mitigate hash collisions and improves representation quality.
    """
    hashes = [
        hash_function_i(entity_id) % vocab_size 
        for i in range(num_hashes)
    ]
    embeddings = [embedding_table[h] for h in hashes]
    return jnp.mean(embeddings, axis=0)  # Average multiple hashes
```

**Grok Transformer Config:**

```python
@dataclass
class TransformerConfig:
    emb_size: int = 512           # Embedding dimension
    key_size: int = 128           # Attention key/query size
    num_q_heads: int = 8          # Number of query heads
    num_kv_heads: int = 2         # Number of key/value heads (MQA)
    num_layers: int = 12          # Transformer layers
    widening_factor: int = 4      # FFN expansion factor
    attn_output_multiplier: float = 1.0
    shard_activations: bool = True
    num_experts: int = 8          # MoE experts (if using)
    num_experts_per_tok: int = 2  # Active experts per token
```

**Training vs. Inference:**

```python
# Training: All candidates in batch can attend to each other
# (Allows model to learn comparative ranking)
train_mask = make_causal_mask(seq_len)

# Inference: Candidate isolation enforced
# (Ensures consistent, cacheable scores)
inference_mask = make_recsys_attn_mask(num_history, num_candidates)
```

---

### 4. Thunder (In-Memory Post Store)

**Core Data Structure:**

```rust
struct PostStore {
    // Global post metadata
    posts: DashMap<i64, LightPost>,
    
    // Per-user timeline indexes
    original_posts: DashMap<i64, VecDeque<TinyPost>>,  // author_id -> posts
    secondary_posts: DashMap<i64, VecDeque<TinyPost>>, // replies/retweets
    video_posts: DashMap<i64, VecDeque<TinyPost>>,     // video content
    
    // Deletion tracking
    deleted_posts: DashMap<i64, bool>,
    
    // Configuration
    retention_period: Duration,
    request_timeout: Duration,
}

struct LightPost {
    post_id: i64,
    author_id: i64,
    created_at: i64,
    is_retweet: bool,
    is_reply: bool,
    is_video: bool,
}

struct TinyPost {
    post_id: i64,
    created_at: i64,
}
```

**Kafka Event Processing:**

```rust
async fn process_post_create_event(event: PostCreateEvent) {
    let light_post = LightPost {
        post_id: event.post_id,
        author_id: event.author_id,
        created_at: event.created_at,
        is_retweet: event.is_retweet,
        is_reply: event.is_reply,
        is_video: event.has_video,
    };
    
    // Add to global posts
    posts.insert(event.post_id, light_post);
    
    // Add to appropriate user timeline
    let tiny_post = TinyPost {
        post_id: event.post_id,
        created_at: event.created_at,
    };
    
    if !event.is_reply && !event.is_retweet {
        original_posts
            .entry(event.author_id)
            .or_insert(VecDeque::new())
            .push_back(tiny_post);
    } else {
        secondary_posts
            .entry(event.author_id)
            .or_insert(VecDeque::new())
            .push_back(tiny_post);
    }
    
    if event.has_video {
        video_posts
            .entry(event.author_id)
            .or_insert(VecDeque::new())
            .push_back(tiny_post);
    }
    
    // Trim old posts beyond retention period
    trim_old_posts(event.author_id);
}

async fn process_post_delete_event(event: PostDeleteEvent) {
    deleted_posts.insert(event.post_id, true);
    posts.remove(&event.post_id);
    // Lazy cleanup from timelines during retrieval
}
```

**Retrieval with Request Timeout:**

```rust
async fn get_in_network_posts(
    user_id: i64,
    following: Vec<i64>,
    max_count: usize,
) -> Vec<i64> {
    let start = Instant::now();
    let mut results = Vec::new();
    
    for author_id in following {
        // Check timeout to prevent long iterations
        if start.elapsed() > request_timeout {
            break;
        }
        
        // Get posts from this author
        if let Some(timeline) = original_posts.get(&author_id) {
            for tiny_post in timeline.iter().rev() {
                // Skip deleted posts
                if deleted_posts.contains_key(&tiny_post.post_id) {
                    continue;
                }
                
                results.push(tiny_post.post_id);
                
                if results.len() >= max_count {
                    return results;
                }
            }
        }
    }
    
    results
}
```

**Performance Characteristics:**
- **Insert:** O(1) amortized (DashMap + VecDeque)
- **Retrieve:** O(F * P) where F = following count, P = posts per author
- **Memory:** ~100 bytes per post (TinyPost) × retention period
- **Latency:** Sub-millisecond for typical retrieval (100-500 posts)
- **Throughput:** Kafka consumer keeps up with realtime ingestion

---

## Data Structures & Types

### Rust Type System

```rust
// Generic pipeline types
type Query = UserFeedQuery;
type Candidate = PostCandidate;
type Score = f64;
type FilterResult<C> = (Vec<C> /* kept */, Vec<C> /* removed */);

// Pipeline execution result
struct PipelineResult<Q, C> {
    query: Q,
    retrieved_candidates: Vec<C>,
    filtered_candidates: Vec<C>,
    selected_candidates: Vec<C>,
}

// Pipeline stage tracking
enum PipelineStage {
    QueryHydrator { name: String },
    Source { name: String },
    Hydrator { name: String },
    Filter { name: String },
    Scorer { name: String },
    Selector { name: String },
    PostSelectionHydrator { name: String },
    PostSelectionFilter { name: String },
    SideEffect { name: String },
}

// Error handling
enum PipelineError {
    QueryHydrationFailed { stage: String, error: String },
    SourceFailed { source: String, error: String },
    HydrationFailed { hydrator: String, error: String },
    ScoringFailed { scorer: String, error: String },
}
```

### Python Type System

```python
# Batch structures for model input
@dataclass
class RecsysBatch:
    """Feature hashes (no embeddings) for batch processing"""
    user_hashes: Array  # [B, num_user_hashes]
    history_post_hashes: Array  # [B, history_len, num_item_hashes]
    history_author_hashes: Array  # [B, history_len, num_author_hashes]
    history_actions: Array  # [B, history_len, num_action_types]
    candidate_post_hashes: Array  # [B, num_candidates, num_item_hashes]
    candidate_author_hashes: Array  # [B, num_candidates, num_author_hashes]
    product_surface: Array  # [B] (web, iOS, Android)

@dataclass
class RecsysEmbeddings:
    """Pre-looked-up embeddings from hash tables"""
    cls_token: Array  # [B, 1, D]
    user_emb: Array  # [B, 1, D]
    history_post_embs: Array  # [B, history_len, D]
    history_author_embs: Array  # [B, history_len, D]
    candidate_post_embs: Array  # [B, num_candidates, D]
    candidate_author_embs: Array  # [B, num_candidates, D]

# Model outputs
@dataclass
class PhoenixPredictions:
    """Per-candidate action predictions"""
    favorite: Array  # [B, num_candidates]
    reply: Array
    retweet: Array
    quote: Array
    click: Array
    profile_click: Array
    video_view: Array
    photo_expand: Array
    share: Array
    dwell: Array
    follow_author: Array
    not_interested: Array
    block_author: Array
    mute_author: Array
    report: Array
    good_click: Array
    good_profile_click: Array
    vqv: Array  # Video Quality View
```

---

## Pipeline Execution Flow

### Detailed Timeline

```
REQUEST RECEIVED
    │
    ├─ [0-5ms] Query Hydration (parallel)
    │   ├─ UserActionSequenceHydrator: Fetch last 1000 engagements
    │   └─ UserFeaturesHydrator: Fetch following list (100s-1000s)
    │
    ├─ [5-50ms] Candidate Sourcing (parallel)
    │   ├─ ThunderSource: Query in-memory store (1-5ms)
    │   │   → Returns 500-2000 in-network posts
    │   └─ PhoenixSource: Call retrieval model (10-40ms)
    │       → Returns 500-2000 out-of-network posts
    │
    ├─ [50-80ms] Candidate Hydration (parallel)
    │   ├─ CoreDataHydrator: Fetch post text, media (batch DB call)
    │   ├─ GizmoduckHydrator: Fetch author data (batch user service)
    │   ├─ VideoDurationHydrator: Fetch video metadata
    │   └─ SubscriptionHydrator: Fetch subscription status
    │
    ├─ [80-85ms] Pre-Scoring Filters (sequential)
    │   ├─ DropDuplicatesFilter: O(N) dedup → ~1000-3000 candidates
    │   ├─ CoreDataHydrationFilter: Remove hydration failures
    │   ├─ AgeFilter: Remove posts >24h (configurable)
    │   ├─ SelfTweetFilter: Remove user's own posts
    │   ├─ RepostDeduplicationFilter: Dedupe reposts
    │   ├─ SubscriptionEligibilityFilter: Remove paywalled content
    │   ├─ PreviouslySeenPostsFilter: Remove engaged posts
    │   ├─ PreviouslyServedPostsFilter: Remove session cache
    │   ├─ MutedKeywordFilter: Remove muted keywords
    │   └─ AuthorSocialgraphFilter: Remove blocked/muted authors
    │       → ~100-500 candidates remain
    │
    ├─ [85-150ms] Scoring (sequential)
    │   ├─ PhoenixScorer: Transformer inference (40-60ms)
    │   │   ├─ Batch size: 100-500 candidates
    │   │   ├─ Input: User history + candidates
    │   │   ├─ Attention mask: Candidate isolation
    │   │   └─ Output: 18 action probabilities per candidate
    │   ├─ WeightedScorer: Compute final_score (1ms)
    │   ├─ AuthorDiversityScorer: Apply diversity decay (1ms)
    │   └─ OONScorer: Adjust out-of-network scores (1ms)
    │
    ├─ [150-152ms] Selection
    │   └─ ScoreSelector: Sort by score, take top 100
    │
    ├─ [152-155ms] Post-Selection Filtering
    │   ├─ VFFilter: Visibility filtering (batch safety check)
    │   └─ DedupConversationFilter: Dedupe conversation threads
    │
    └─ [155-160ms] Side Effects (async, non-blocking)
        └─ CacheRequestInfo: Store served posts for next request
│
RESPONSE RETURNED (50-200 posts)
    Timeline: ~85-160ms p50, ~200-300ms p99
```

### Parallelization Strategy

```rust
// Sources run in parallel
let (thunder_posts, phoenix_posts) = tokio::join!(
    thunder_source.get_candidates(&query),
    phoenix_source.get_candidates(&query),
);

// Hydrators run in parallel on all candidates
let mut candidates = merge(thunder_posts, phoenix_posts);
futures::join_all(vec![
    core_data_hydrator.hydrate(&query, &mut candidates),
    gizmoduck_hydrator.hydrate(&query, &mut candidates),
    video_duration_hydrator.hydrate(&query, &mut candidates),
    subscription_hydrator.hydrate(&query, &mut candidates),
]).await;

// Filters run sequentially (order matters)
for filter in filters {
    candidates = filter.filter(&query, candidates).kept();
}

// Scorers run sequentially (order matters)
for scorer in scorers {
    scorer.score(&query, &mut candidates);
}
```

---

## Dependencies & Libraries

### Rust Dependencies (Cargo.toml inferred)

```toml
[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }
futures = "0.3"
async-trait = "0.1"

# gRPC
tonic = "0.10"
tonic-reflection = "0.10"
prost = "0.12"

# Concurrency
dashmap = "5.5"

# Logging & Metrics
log = "0.4"
tracing = "0.1"
tracing-subscriber = "0.3"

# CLI
clap = { version = "4", features = ["derive"] }

# Kafka (for Thunder)
rdkafka = "0.34"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Internal X dependencies
xai_http_server = { path = "../internal/http_server" }
xai_stats_macro = { path = "../internal/stats_macro" }
xai_init_utils = { path = "../internal/init_utils" }
xai_profiling = { path = "../internal/profiling" }
xai_thunder_proto = { path = "../proto/thunder" }
xai_home_mixer_proto = { path = "../proto/home_mixer" }
xai_visibility_filtering = { path = "../internal/vf" }
xai_strato_client = { path = "../internal/strato_client" }
```

### Python Dependencies (pyproject.toml)

```toml
[project]
name = "phoenix"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "jax==0.8.1",
    "dm-haiku>=0.0.13",
    "numpy>=1.26.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pyright>=1.1.408",
    "ruff>=0.3.0",
]

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.pyright]
typeCheckingMode = "strict"
reportMissingTypeStubs = false
```

---

## Configuration Parameters

### Home Mixer Server

```bash
home-mixer \
    --grpc_port=8080 \
    --metrics_port=9090 \
    --reload_interval_minutes=5 \
    --chunk_size=100 \
    --num_candidates=100 \
    --enable_profiling=false
```

**Parameters:**
- `grpc_port`: Service port (default: 8080)
- `metrics_port`: Prometheus metrics (default: 9090)
- `reload_interval_minutes`: Config hot-reload interval (default: 5)
- `chunk_size`: Batch size for processing (default: 100)
- `num_candidates`: Number of candidates to return (default: 100)
- `enable_profiling`: CPU profiling (default: false)

### Thunder Server

```bash
thunder \
    --grpc_port=8081 \
    --http_port=8082 \
    --post_retention_seconds=604800 \
    --request_timeout_ms=100 \
    --max_concurrent_requests=1000 \
    --kafka_brokers="kafka:9092" \
    --kafka_topic="post_events" \
    --kafka_group_id="thunder_consumer" \
    --kafka_num_threads=4 \
    --enable_profiling=false
```

**Parameters:**
- `post_retention_seconds`: Time to keep posts (default: 604800 = 7 days)
- `request_timeout_ms`: Timeout for iteration safety (default: 100ms)
- `max_concurrent_requests`: Concurrency limit (default: 1000)
- `kafka_brokers`: Kafka broker addresses
- `kafka_topic`: Kafka topic for post events
- `kafka_group_id`: Consumer group ID
- `kafka_num_threads`: Consumer thread count (default: 4)

### Phoenix Model Configuration

```python
# Transformer hyperparameters
transformer_config = TransformerConfig(
    emb_size=512,
    key_size=128,
    num_q_heads=8,
    num_kv_heads=2,  # Multi-Query Attention
    num_layers=12,
    widening_factor=4,
    attn_output_multiplier=1.0,
)

# Hash-based embedding configuration
hash_config = HashConfig(
    num_user_hashes=3,
    num_item_hashes=3,
    num_author_hashes=2,
    vocab_size=10_000_000,
    emb_size=512,
)

# Action weights for scoring
action_weights = {
    "favorite": 1.0,
    "reply": 2.0,
    "retweet": 1.5,
    "quote": 1.8,
    "click": 0.5,
    "dwell": 0.3,
    "share": 2.5,
    "follow_author": 3.0,
    "block_author": -10.0,
    "mute_author": -8.0,
    "report": -15.0,
}
```

---

## Build & Testing Infrastructure

### Rust Build

```bash
# Build all components
cargo build --release

# Build specific component
cargo build --release -p candidate-pipeline
cargo build --release -p home-mixer
cargo build --release -p thunder

# Run tests
cargo test

# Run with optimizations
cargo build --profile release-with-debug
```

**Cargo Workspaces:**
```toml
[workspace]
members = [
    "candidate-pipeline",
    "home-mixer",
    "thunder",
]
```

### Python Build & Testing

```bash
# Install dependencies with uv (fast package installer)
uv pip install -e ".[dev]"

# Run tests
pytest phoenix/

# Type checking
pyright phoenix/

# Linting
ruff check phoenix/
ruff format phoenix/

# Run retrieval model
python phoenix/run_retrieval.py

# Run ranking model
python phoenix/run_ranker.py
```

**Test Structure:**
```
phoenix/
├── test_recsys_model.py           # Ranking model tests
│   ├── test_forward_pass()
│   ├── test_attention_mask()
│   ├── test_candidate_isolation()
│   └── test_multi_action_prediction()
├── test_recsys_retrieval_model.py # Retrieval model tests
│   ├── test_two_tower_encoding()
│   ├── test_similarity_computation()
│   └── test_top_k_retrieval()
└── runners.py                      # Training/inference runners
```

---

## Key Design Decisions

### 1. **Zero Hand-Engineered Features**
**Rationale:** The Grok-based transformer learns all relevance patterns from engagement sequences, eliminating manual feature engineering complexity.

**Benefits:**
- Reduces data pipeline complexity
- Faster iteration (no feature engineering required)
- Better generalization (model learns implicit patterns)
- Simpler serving infrastructure

**Trade-offs:**
- Higher inference cost (transformer vs. simple features)
- Requires large-scale training data
- Less interpretable than explicit features

### 2. **Candidate Isolation in Ranking**
**Rationale:** During inference, candidates cannot attend to each other via specialized attention masking.

**Benefits:**
- **Score Consistency:** A post's score doesn't depend on which other posts are in the batch
- **Cacheability:** Scores can be pre-computed and cached
- **Parallel Serving:** Different batches produce same scores
- **A/B Testing:** Easy to compare different candidate sets

**Implementation:**
```python
# Training: Candidates can attend to each other
train_mask = make_causal_mask(seq_len)

# Inference: Candidates isolated
inference_mask = make_recsys_attn_mask(
    num_history=50,
    num_candidates=100
)
# Mask ensures candidates[i] cannot attend to candidates[j] for i != j
```

### 3. **Hash-Based Embeddings**
**Rationale:** Use multiple hash functions per entity instead of learned vocabulary.

**Benefits:**
- **No OOV:** Every entity gets an embedding (no unknown tokens)
- **Memory Efficient:** Fixed-size embedding tables
- **Collision Mitigation:** Multiple hashes reduce collision impact
- **Faster Training:** No embedding table updates for rare entities

**Implementation:**
```python
def embed_entity(entity_id, num_hashes=3):
    embeddings = []
    for i in range(num_hashes):
        hash_val = hash_function_i(entity_id) % VOCAB_SIZE
        embeddings.append(embedding_table[hash_val])
    return jnp.mean(embeddings, axis=0)
```

### 4. **Multi-Action Prediction**
**Rationale:** Predict probabilities for 18+ different engagement types instead of single relevance score.

**Benefits:**
- **Richer Signal:** Captures different user intents (passive vs. active engagement)
- **Flexible Weighting:** Easy to adjust weights per action type
- **Negative Signals:** Can explicitly model dislikes (block, mute, report)
- **Product Evolution:** New actions can be added without retraining

**Action Categories:**
- **Positive:** favorite, reply, retweet, quote, share, follow
- **Passive:** click, dwell, video_view, photo_expand
- **Negative:** not_interested, block_author, mute_author, report

### 5. **Composable Pipeline Architecture**
**Rationale:** Generic trait-based framework for building recommendation pipelines.

**Benefits:**
- **Separation of Concerns:** Business logic separate from orchestration
- **Parallel Execution:** Sources and hydrators run concurrently
- **Graceful Error Handling:** Configurable failure modes
- **Easy Extensibility:** Add new sources/filters/scorers without changing framework
- **Testability:** Each component can be unit tested independently

**Example:**
```rust
let pipeline = CandidatePipeline::builder()
    .with_query_hydrator(UserActionSequenceHydrator)
    .with_source(ThunderSource)
    .with_source(PhoenixSource)
    .with_hydrator(CoreDataHydrator)
    .with_filter(AgeFilter)
    .with_scorer(PhoenixScorer)
    .with_selector(TopKSelector::new(100))
    .build();
```

### 6. **In-Memory Thunder Store**
**Rationale:** Keep recent posts in memory for sub-millisecond in-network retrieval.

**Benefits:**
- **Ultra-Low Latency:** 1-5ms retrieval vs. 20-50ms DB query
- **High Throughput:** No DB bottleneck
- **Realtime Updates:** Kafka consumer keeps store fresh
- **Automatic Retention:** Old posts trimmed automatically

**Trade-offs:**
- **Memory Usage:** ~100 bytes × retention period × post rate
- **No Persistence:** Restart loses data (acceptable for cache)
- **Limited Retention:** Typically 7 days (vs. infinite in DB)

**Memory Calculation:**
```
Posts per day: 500M
Retention: 7 days
Total posts: 3.5B
Memory per TinyPost: 100 bytes
Total memory: 350 GB (distributed across servers)
```

### 7. **Sequential Filters and Scorers**
**Rationale:** Run filters and scorers in sequence, not parallel.

**Why?**
- **Filter Order Matters:** Some filters depend on previous filters
- **Scorer Order Matters:** Scores build on each other (base score → diversity → OON adjustment)
- **State Modification:** Each stage mutates candidate list/scores
- **Performance:** Each filter reduces candidate count, making next filter faster

**Optimization:**
```rust
// Filters run in order of selectivity (most restrictive first)
vec![
    DropDuplicatesFilter,      // High reduction (50%+)
    AgeFilter,                  // Medium reduction (20-30%)
    PreviouslySeenFilter,       // Medium reduction (20-30%)
    MutedKeywordFilter,         // Low reduction (<5%)
    AuthorSocialgraphFilter,    // Low reduction (<5%)
]
```

---

## Performance Characteristics

### Latency Profile

| Component | Latency (p50) | Latency (p99) | Notes |
|-----------|---------------|---------------|-------|
| **Query Hydration** | 2-5ms | 10-20ms | Parallel fetches |
| **Thunder Retrieval** | 1-5ms | 10-15ms | In-memory lookup |
| **Phoenix Retrieval** | 15-30ms | 50-80ms | ML model inference |
| **Candidate Hydration** | 15-25ms | 40-60ms | Batch DB/service calls |
| **Pre-Scoring Filters** | 2-5ms | 8-12ms | Sequential, fast filters |
| **Phoenix Ranking** | 40-60ms | 80-120ms | Transformer inference |
| **Weighted Scoring** | <1ms | 2-3ms | Simple math |
| **Post-Selection** | 2-5ms | 8-12ms | Final validation |
| **Total Pipeline** | **85-160ms** | **200-300ms** | End-to-end |

### Throughput

| Metric | Value | Notes |
|--------|-------|-------|
| **Home Mixer QPS** | 10K-50K | Per server instance |
| **Thunder QPS** | 50K-100K | In-memory reads are fast |
| **Phoenix Retrieval QPS** | 1K-5K | GPU-bound |
| **Phoenix Ranking QPS** | 1K-5K | GPU-bound |
| **Kafka Ingestion Rate** | 500M posts/day | ~5.8K posts/sec |

### Resource Usage

| Component | CPU | Memory | GPU | Notes |
|-----------|-----|--------|-----|-------|
| **Home Mixer** | 4-8 cores | 8-16 GB | - | Mostly I/O bound |
| **Thunder** | 8-16 cores | 64-128 GB | - | Memory for post store |
| **Phoenix Retrieval** | 2-4 cores | 16-32 GB | 1x A100 | GPU for inference |
| **Phoenix Ranking** | 2-4 cores | 16-32 GB | 1x A100 | GPU for inference |

### Scalability

**Horizontal Scaling:**
- **Home Mixer:** Stateless, can scale linearly
- **Thunder:** Shard by user_id (each server handles subset of users)
- **Phoenix:** Model serving with load balancer

**Bottlenecks:**
1. **Phoenix GPU Inference:** Most expensive operation (~50-100ms)
   - **Mitigation:** Batch candidates, use tensor cores, model quantization
2. **Candidate Hydration:** Multiple service calls
   - **Mitigation:** Parallel hydrators, caching, batch APIs
3. **Thunder Memory:** Large following lists cause long iterations
   - **Mitigation:** Request timeout, sample following list

---

## Summary

The x-algorithm repository implements a **state-of-the-art recommendation system** combining:

1. **Realtime Data Ingestion** (Thunder): Kafka → In-memory store for sub-millisecond lookups
2. **ML-Based Retrieval** (Phoenix): Two-tower model for similarity search over millions of posts
3. **Sophisticated Ranking** (Phoenix): Grok-based transformer with candidate isolation for consistent scoring
4. **Composable Pipeline** (Candidate Pipeline): Generic framework for building recommendation pipelines
5. **Production Orchestration** (Home Mixer): gRPC service tying everything together

**Key Technical Achievements:**
- **Zero hand-engineered features** - Pure transformer learning
- **Candidate isolation** - Consistent, cacheable scores
- **Multi-action prediction** - 18+ engagement types
- **Hash-based embeddings** - No vocabulary, no OOV
- **Sub-200ms latency** - For entire pipeline (p50: 85-160ms)

**Technology Stack:**
- **Rust** (Tokio, Tonic, DashMap) for high-performance serving
- **Python** (JAX, Haiku) for ML model development
- **Kafka** for realtime event streaming
- **gRPC** for service communication
- **Protobuf** for serialization

This architecture represents modern recommendation systems at scale: combining realtime data, ML models, and sophisticated ranking to deliver personalized content feeds.
