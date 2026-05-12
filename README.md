# Origin AI Engineering Take-Home: Referral Inbox Triage Agent

Origin builds software for pediatric therapy practices. In this assignment, you are helping a fictional practice, Cedar Kids Therapy, triage its Monday inbox.

## Scenario

It is Monday at 8am at a multi-disciplinary pediatric therapy practice supporting speech-language pathology, occupational therapy, and physical therapy. The shared inbox accumulated items over the weekend from pediatrician fax referrals, parent voicemails, parent portal messages, and emails. Build an AI agent prototype that turns the messy batch into a sorted, human-reviewable action plan.

## How to Run

```bash
npm install
export ANTHROPIC_API_KEY=your_key_here
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

## Stack and Runtime

TypeScript, Node LTS, `@anthropic-ai/sdk` (claude-sonnet-4-5). Triage runs in roughly 15–20 seconds for 8 items — one parallel LLM call per item for classification and extraction, one for draft generation. Built with Claude Code as a coding assistant.

## Architecture

Each inbox item is processed in parallel via `Promise.allSettled`. Per item:

**1. `analyzeItem(item)`** — one LLM call using tool_use with a strict JSON schema. Returns both a classification signal and all extracted intake fields (child name, DOB, payer, member ID, contacts, discipline, scheduling preferences). Tool use is used instead of prompting for JSON to guarantee structured output — no markdown fences, no parsing errors. Temperature 0 for determinism. Result is merged with an all-null fallback object so missing fields are always `null` rather than `undefined`, which keeps schema validation clean.

**2. Router** — a switch on the classification signal dispatches to the appropriate handler: `safeguarding`, `same_day_cancellation`, `incomplete_referral`, `clinical_question`, `new_referral`, or `other`. Safeguarding is checked first and unconditionally in the LLM prompt — any hint of harm overrides all other signals.

**3. Handlers** — each handler calls tools in a deliberate sequence driven by policy logic, not speculatively. Key decisions:
- `verify_insurance` is called before any `find_slots` or `hold_slot`. Out-of-network and expired coverage branch immediately to a billing task with no slot hold.
- Billing system result supersedes the referral document — discrepancies are surfaced in `missing_info` and `decision_rationale`.
- Spanish-speaking families trigger `lookup_policy("language_access")` and `find_slots` with `language: "es"` to match a Spanish-capable provider.
- `search_patient` is called on all referrals with enough data to attempt a match — existing patients are classified as `existing_patient_request`.

**4. `generateDraft()`** — a second LLM call with a tightly constrained prompt: exactly 2 sentences, explicit format rules, concrete bad examples to pattern-match against, temperature 0.3, max_tokens 120. Failures fall back to a template string silently.

**5. Error resilience** — `Promise.allSettled` with a `fallbackOutput` stub ensures every item produces valid output even if one handler throws. No item fails silently.

All triage routing, urgency assignment, and tool sequencing is rule-based and deterministic. LLM is used only where language understanding is genuinely required: extracting structured fields from unstructured voicemails and emails, and generating human-facing draft replies.

## Failure Modes and Production Eval

**Over-escalation** is guarded against — the router defaults to P2 and only escalates on clear signals. The LLM classification prompt instructs safeguarding to win on ambiguity, which is the correct asymmetry for a clinical setting where false negatives are more dangerous than false positives.

**LLM extraction failures** fall back to an all-null `ItemAnalysis` via `{ ...fallback, ...toolUse.input }` merge — schema validation stays clean. Draft failures fall back to template strings. No item fails silently.

**Out-of-network and expired insurance** are detected before any slot hold — the agent never holds a slot for a family that needs a benefits conversation first.

**Billing system supersedes referral documents** — when `verify_insurance` returns a payer status that conflicts with the referral document, the discrepancy is explicitly surfaced in `missing_info` and `decision_rationale` per policy.

**Incomplete referrals** are routed to intake with a follow-up task rather than attempting to proceed with missing fields.

In production I would add: golden-set evals across item types with variance tracking across runs, structured output validation on draft content (classifier to confirm no clinical advice or scheduling language leaked through), latency budgets per item with alerting, and a dry-run mode that skips `hold_slot` for CI pipelines.

## What I Chose Not to Build, and Why

**LLM-based urgency scoring** — all urgency signals are deterministic from policy rules. Using an LLM to assign P0/P1/P2 would introduce non-determinism into safety-critical routing decisions. Rule-based urgency is the correct choice here.

**Per-provider slot matching on free-text preferences** — the agent passes scheduling preferences to `find_slots` but does not parse "after school Tuesdays" into structured time filters. The tool interface does not support that granularity and the marginal value did not justify the complexity within the time box.

**Patient record creation** — `search_patient` is called for all referrals with sufficient data. New patients are referenced by name string in tool calls. Creating records is out of scope for a triage agent.

**Automated rescheduling** — `find_slots` and `hold_slot` are called but no appointment is ever confirmed. This is an explicit constraint and the correct production behavior — scheduling requires human sign-off.

## What I Would Do With Another 4 Hours

**Golden-set eval harness** — synthetic variants covering: dual safeguarding + referral in one message, expired insurance, no matching provider for the requested discipline, non-English referral without a Spanish-capable provider, existing patient with an insurance conflict between the referral doc and billing system.

**Structured draft validation** — run a lightweight classifier over each generated draft before it enters the output to confirm no clinical advice and no scheduling language leaked through. Currently the constraints are enforced by prompt only.

**Domain-specific NER model** — the current extraction relies on a general-purpose LLM call per item. With sufficient historical intake data, a fine-tuned NER model (e.g. GLiNER on DeBERTa-v3) trained on clinic-specific entities — child name, parent contact, payer, member ID, discipline — would be faster, cheaper, and more consistent than a prompted LLM. This is the same approach used in the GLiNER Sports NER project in my portfolio, where a fine-tuned SLM matched GPT-4o-mini F1 at 10x lower latency and zero per-document cost.

**Confidence field on classification** — surface low-confidence items for human review even when the signal looks clean. Currently the agent is binary — it either classifies or falls to `other`. A confidence score would let staff prioritize their review queue.

**Retry logic on `analyzeItem`** — currently a single API failure falls back to all-null which routes to `handleOther`. A retry with exponential backoff would recover most transient failures before giving up.