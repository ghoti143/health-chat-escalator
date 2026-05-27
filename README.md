# health-chat-escalator

Classifies inbound patient chat messages and decides whether to escalate to a clinician immediately.

## Setup

```bash
npm install
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Usage

```typescript
import { classifyMessage } from "./src/classifier.js";

const decision = await classifyMessage(
  "I've had crushing chest pain for 30 minutes radiating to my left arm.",
);
// { escalate: true, category: "cardiac", confidence: "high", reasoning: "..." }
```

## Run tests

```bash
npm test
```

Tests hit the live Claude API — expect ~30–60 s per test, 90 s timeout each. Set `ANTHROPIC_API_KEY` before running.

## Design notes

- **Model**: `claude-opus-4-7` with adaptive thinking for safety-critical classification
- **Prompt caching**: system prompt is marked `cache_control: ephemeral` — repeat calls avoid re-sending ~1 600 tokens
- **Structured output**: `messages.parse()` + `zodOutputFormat()` enforces the response schema at the SDK level
- **Error handling**: any failure (network, parse error, unexpected shape) returns `{ escalate: true, category: "unknown", confidence: "low", ... }` — never silently defaults to non-escalation

## EscalationDecision

```typescript
type EscalationDecision = {
  escalate: boolean;
  category: "cardiac" | "suicidal_ideation" | "hypoglycemia" | "dka" | "other_urgent" | null;
  confidence: "high" | "medium" | "low";
  reasoning: string;
};
```
