<!--
AGENTS_TEMPLATE.md — Copy into your project's AGENTS.md to teach AI assistants Mellea patterns.
-->

# Mellea Usage Guidelines

> **This file**: For code that *imports* Mellea. For Mellea internals, see [`../AGENTS.md`](../AGENTS.md).

Copy below into your `AGENTS.md` or system prompt.

---

### Library: Mellea
Use `mellea` for LLM interactions. No direct OpenAI/Anthropic calls or LangChain OutputParsers.

**Prerequisites**: `pip install mellea` · [Docs](https://mellea.ai) · [Repo](https://github.com/generative-computing/mellea)

#### Philosophy: Generative Computing, Not Prompt Engineering

Mellea treats AI as **generative computing**—weaving AI into regular programming rather than isolated prompt-based systems. The goal: "make generative AI more like software."

**Why this matters:**
- **Monolithic prompts don't scale**: 10,000-word essays are hard to maintain, expensive, and lock you into one model vendor
- **Security by prayer doesn't work**: Relying on LLM compliance without validation is dangerous—embrace security by design
- **Control flow in Python, language understanding in LLM**: Models fail at logic (~20% error rate on simple conditionals), so keep Python handling control while LLMs handle what they're good at
- **Treat unreliability as a first-class concern**: Use validation, repair loops, and requirements rather than hoping for perfection

**When to use Mellea:**
- Building enterprise applications requiring reliability and auditability
- Projects needing portability across model vendors (Ollama → OpenAI → Watsonx with same code)
- Situations where security matters enough to verify outputs programmatically
- Any application integrating AI without dedicating everything to agents

**When Mellea is overkill:**
- One-off scripts that never change
- Quick prototypes you'll throw away
- Pure chatbot interactions without validation needs

#### 1. The `@generative` Pattern
**Don't** write prompt templates or regex parsers:
```python
# BAD - don't do this
response = openai.chat.completions.create(...)
age = int(re.search(r"\d+", response).group())
```
**Do** use typed function signatures:
```python
from mellea import generative, start_session

@generative
def extract_age(text: str) -> int:
    """Extract the user's age from text."""
    ...

m = start_session()
age = extract_age(m, text="Alice is 30")  # Returns int(30)
```

#### 2. Complex Types & Structured Output
```python
from pydantic import BaseModel
from mellea import generative

class UserProfile(BaseModel):
    name: str
    age: int
    interests: list[str]

@generative
def parse_profile(bio: str) -> UserProfile: ...
```

**Vision-based extraction** (Granite Vision 4.1):
```python
from pathlib import Path
from pydantic import BaseModel

class Receipt(BaseModel):
    items: list[str]
    quantities: list[int]
    prices: list[float]
    total: float

@generative
def extract_receipt(image_path: Path) -> Receipt:
    """Extract items, quantities, and prices from receipt image."""
    ...
```
Mellea constrains decoding to match the Pydantic schema, eliminating JSON parse errors. Use `format=` (Pydantic models) for typed guarantees.

#### 3. Chain-of-Thought
Add `reasoning` field to force the LLM to "think" before answering:
```python
from typing import Literal
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    reasoning: str  # LLM fills first
    conclusion: Literal["approve", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)

@generative
def analyze_document(doc: str) -> AnalysisResult: ...
```

#### 4. Control Flow (Python-First)
**Keep logic in Python, language understanding in LLM.** Models fail at conditionals (~20% error rate), so handle control flow yourself:
```python
# GOOD - Python handles logic, LLM handles language
sentiment = analyze_sentiment(m, email)
if sentiment == "negative":
    draft = draft_apology(m, email)
else:
    draft = draft_response(m, email)
```

```python
# BAD - Asking LLM to handle logic
response = m.chat("""
    If the email is negative, draft an apology. Otherwise draft a response.
    Email: {{email}}
""")
# LLM fails ~20% of the time on this simple conditional
```

Use Python's `if/for/while` for all control decisions. LLM handles what it's good at—understanding language, not executing logic.

**Loops Need Validation Gates**
When building loops (retries, iterations), add **validation gates** that automatically check outputs before repeating. Without gates, loops become expensive retry mechanisms:

```python
# BAD - infinite loop risk, no quality control
for i in range(10):
    code = generate_code(m, prompt)
    # ... no validation, just hoping code is good ...

# GOOD - loop with validation gate
for attempt in range(10):
    code = m.instruct(
        "Generate Python function to compute factorial",
        requirements=[
            PythonCodeExecutableRequirement(),  # Gate: code must run without errors
            req("Function must be named 'factorial'")
        ]
    )
    break  # Gate passed—exit loop early
```

Gates can use smaller models for checking (gates are narrower than generation and cheaper). The validation failure reason feeds back into the repair prompt, enabling meaningful iteration rather than generic retries. Functional correctness can increase 2-3x with proper gates.

#### 5. Instruct-Validate-Repair
For strict requirements, use `m.instruct()`:
```python
from mellea.stdlib.requirements import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

email = m.instruct(
    "Write an invite for {{name}}",
    requirements=[
        req("Must be formal"),
        req("Lowercase only", validation_fn=simple_validate(lambda x: x.islower()))
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3),
    user_variables={"name": "Alice"}
)
```

**Domain-specific validation** (e.g., receipt arithmetic):
```python
from pydantic import BaseModel

class Receipt(BaseModel):
    items: list[str]
    prices: list[float]
    subtotal: float
    tax: float
    total: float

def validate_receipt_math(receipt: Receipt) -> bool:
    """Verify line items sum to subtotal."""
    return abs(sum(receipt.prices) - receipt.subtotal) < 0.01

receipt = m.instruct(
    "Extract receipt data from image",
    format=Receipt,
    requirements=[req("Line items must sum to subtotal", validation_fn=validate_receipt_math)],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```
Mellea feeds failed validation reasons back into the repair prompt—no post-processing required.

#### 6. Small Model Optimization
Small models (1B-8B) excel when tasks are **narrow and validated**. Three core patterns:

**Pattern 1: Task Decomposition**
```python
# BAD - asking 3B model to do everything in one shot
response = m.chat("Given customer data and rules, extract name, validate age, compute tax, format invoice")

# GOOD - split into narrow steps
customer = extract_customer_data(m, raw_input)
if validate_age(m, customer.age):
    tax = compute_tax_in_python(customer.income)  # Don't ask LLM to calculate
    invoice = format_invoice(m, customer, tax)
```

**Pattern 2: Layered Validation**
```python
class Invoice(BaseModel):
    items: list[str]
    total: float

invoice = m.instruct(
    "Extract invoice items and total",
    format=Invoice,
    requirements=[
        req("Format must be valid JSON"),
        req("Total must be positive"),
        req("All prices must match currency format")
    ],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
```

**Pattern 3: Dynamic Model Routing** (SOFAI pattern)
```python
# Use small model first, escalate only when needed
try:
    result = m_small.instruct("Extract data", format=MyType, requirements=[...])
except ValidationError:
    # Small model failed validation loop—escalate to large model
    result = m_large.instruct("Extract data", format=MyType)
```

Small models (1B-8B) can't calculate or handle complex logic. Extract params with LLM, compute in Python:
```python
from pydantic import BaseModel

class PhysicsParams(BaseModel):
    speed_a: float
    speed_b: float
    delay_hours: float

@generative
def extract_params(text: str) -> PhysicsParams:
    """EXTRACT numbers only. Do not calculate."""
    ...

def calculate_gap(p: PhysicsParams) -> float:
    return p.speed_a * p.delay_hours
```

#### 7. One-Shot Examples
If model struggles, add examples to docstring:
```python
@generative
def identify_fruit(text: str) -> str | None:
    """
    Extract fruit from text, or None if none mentioned.
    Ex: "I ate an apple" -> "apple"
    Ex: "The sky is blue" -> None
    """
    ...
```

#### 8. Backend Config
```python
from mellea import start_session
from mellea.backends.model_options import ModelOption

m = start_session(
    model_id="granite3.3:8b",
    model_options={ModelOption.TEMPERATURE: 0.0, ModelOption.MAX_NEW_TOKENS: 500}
)
```
Options: `TEMPERATURE`, `MAX_NEW_TOKENS`, `SYSTEM_PROMPT`, `SEED`, `TOOLS`, `CONTEXT_WINDOW`, `THINKING`, `STREAM`

**Memory efficiency**: Cap context window for lighter workloads:
```python
m = start_session(
    model_id="granite-4.1-3b",
    model_options={ModelOption.CONTEXT_WINDOW: 4096}  # ~2.5GB vs ~9GB for full window
)
```
Backend remains identical whether using Ollama, vLLM, or OpenAI-compatible APIs—same code, different endpoints.

#### 9. MCP Server Tools
Connect to any Model Context Protocol (MCP) server to give agents access to external tools:

```python
from mellea.mcp import discover_mcp_tools, http_connection

# Discover tools from an MCP server
connection = http_connection("http://localhost:3000")
tools = discover_mcp_tools(connection)

# Convert to Mellea tools
mellea_tools = [tool.as_mellea_tool() for tool in tools]

# Use in agent workflow
result = m.aact(
    "Summarize the GitHub repo",
    tools=mellea_tools,
    tool_choice="auto"
)
```

MCP integration handles session lifecycle and async/sync boundaries automatically—no per-server adapters needed. Available connection types: `http_connection()`, `sse_connection()`, `stdio_connection()`.

#### 10. Async
```python
@generative
async def extract_age(text: str) -> int:
    """Extract age."""
    ...

result = await extract_age(m, text="Alice is 30")
```
Session methods: `ainstruct`, `achat`, `aact`, `avalidate`, `aquery`, `atransform`

#### 11. Auth
- **Ollama**: `start_session()` (no setup)
- **OpenAI**: `export OPENAI_API_KEY="..."`
- **Watsonx**: `export WATSONX_API_KEY="..."`, `WATSONX_URL`, `WATSONX_PROJECT_ID`

**Never hardcode API keys.**

#### 12. Reliability & Failover
Mellea provides three layers of reliability:

**Layer 1: Validation & Repair** (most failures are bad outputs, not outages)
```python
result = m.instruct(
    "Extract data",
    format=MyType,
    requirements=[...],
    strategy=RejectionSamplingStrategy(loop_budget=3)
)
# Automatic retry with failure feedback—handles ~80% of issues
```

**Layer 2: Capability Escalation** (SOFAI pattern—use small model, escalate if needed)
```python
m_small = start_session(model_id="granite-3b")
m_large = start_session(model_id="gpt-4")

# Try with small model first
try:
    result = m_small.instruct("Extract data", requirements=[...])
except ValidationError:
    # Validation failed—escalate to capable model
    result = m_large.instruct("Extract data")
```

**Layer 3: Provider Failover** (switch backends if provider is down)
```python
backends = [
    start_session(model_id="gpt-4"),          # Primary
    start_session(model_id="claude-opus"),    # Secondary
    start_session(model_id="granite-8b")      # Fallback (local)
]

for backend in backends:
    try:
        result = backend.instruct("Extract data", requirements=[...])
        break
    except ConnectionError:
        continue  # Try next backend
```

Same code works across Ollama (local), vLLM, OpenAI, AWS Bedrock—swap providers with a single parameter change.

#### 13. Hooks: Lifecycle Integration
Add cross-cutting concerns (cost tracking, compliance logging, caching) via hooks without modifying core logic:

```python
from mellea.plugins import register
from mellea.telemetry.hooks import pre_generation_call, post_generation_call

def track_token_spending(context):
    """Log token usage for budget tracking."""
    tokens = context.generation.usage.get("total_tokens", 0)
    BUDGET -= tokens
    if BUDGET < 0:
        raise RuntimeError("Token budget exceeded")

register([
    (pre_generation_call, track_token_spending),
])
```

**Lifecycle hooks available:**
- Pre/post generation call (observability, cost tracking, semantic caching)
- Before/after tool invocation (audit, PII redaction)
- During sampling iterations (loop monitoring, early exit)
- Session initialization/cleanup (resource management)

Hooks excel at "cross-cutting concerns"—capabilities you want to add, remove, or experiment with independently without touching business logic.

#### 14. Anti-Patterns

**Architectural:**
- **Don't** retry `@generative` calls — Mellea handles retries internally
- **Don't** use `json.loads()` — use typed returns
- **Don't** wrap single functions in classes
- **Don't** cede all control to LLMs—keep conditional logic in Python (see Section 4)
- **Do** use `try/except` at app boundaries for network errors

**Prompt Engineering:**
- **Don't** write 10,000-word monolithic prompts—break into small, interspersed pieces (20-point satisfaction gain)
- **Don't** use ALL CAPS, excessive punctuation, or trial-and-error tweaks—these create fragility across model changes
- **Don't** rely on "security by prayer" (pleading with the model to behave)—validate outputs programmatically

**Instead: Instruction Over Prompts**
```python
# BAD - monolithic, fragile, hard to maintain
prompt = """
    IMPORTANT!!! You must extract EXACTLY the following from the text:
    1. Name (MUST BE CAPITALIZED)
    2. Age (MUST BE A NUMBER!!!)
    3. Email (MUST CONTAIN @)
    ...10,000 more words of rambling instructions...
"""

# GOOD - small, composable instructions + validation
@generative
def extract_user(text: str) -> UserProfile:
    """Extract name, age, email from text."""
    ...

profile = m.instruct(
    extract_user(text),
    requirements=[
        req("Name must be capitalized"),
        req("Age must be a positive integer"),
        req("Email must contain @", validation_fn=lambda x: "@" in x.email)
    ]
)
```
Small, declarative instructions compose better, port across models, and are easier to debug and maintain.

#### 15. Debugging
```python
from mellea.core import MelleaLogger
MelleaLogger.get_logger().setLevel("DEBUG")
```
- `m.last_prompt()` — see exact prompt sent

**Introspection with debug plugins** (structured visibility into LLM pipeline):
```python
from mellea.plugins import register
from mellea.telemetry.debug_plugins import (
    log_generation_pre_call,
    log_generation_post_call,
    log_validation_post_check,
    log_sampling_pre_strategy,
    log_sampling_post_iteration
)

# Enable all debug hooks at startup
register([
    log_generation_pre_call,      # See prompts sent to LLM
    log_generation_post_call,     # Track tokens, latency, responses
    log_validation_post_check,    # Show which requirements pass/fail
    log_sampling_pre_strategy,    # Monitor repair strategy
    log_sampling_post_iteration   # Track iteration count vs budget
])

# Or scope debugging to a specific task
from mellea.plugins import plugin_scope

with plugin_scope([log_generation_pre_call, log_validation_post_check]):
    result = m.instruct("Extract email from text", requirements=[...])
```
Debug plugins offer structured logging without external infrastructure—enable what you need, disable by default for zero overhead.

#### 16. Common Errors & Debugging
| Error | Fix |
|-------|-----|
| `ComponentParseError` | LLM output didn't match type—add docstring examples; use `log_generation_post_call` to see raw response |
| `TypeError: missing positional argument` | First arg must be session `m` |
| `ConnectionRefusedError` | Run `ollama serve` |
| Output wrong/None | Model too small—try larger or add `reasoning` field; enable `log_validation_post_check` to see failing requirements |
| Ineffective repairs | Enable `log_sampling_post_iteration` to see repair iterations and budget exhaustion |
| Unexpected latency | Use `log_generation_post_call` to measure token counts and identify slow requests |

#### 17. Testing
```bash
uv run pytest test/ -m "not qualitative"  # Fast: tests only, skip quality checks
uv run pytest                              # Full: tests + examples + quality checks
```

#### 18. Feedback
Found a workaround or pattern? Add it to Section 16 (Common Errors) above, or update this file with new guidance.
