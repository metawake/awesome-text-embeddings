# Contributing

This list is **curated, not comprehensive**. We keep signal high so developers can find what they need fast.

## What Belongs Here

**✅ In scope:** Embedding models, benchmarking tools, fine-tuning frameworks, foundational papers

**❌ Out of scope:** Vector databases, RAG frameworks, general ML tools → check [Related Lists](README.md#related-lists)

## The Curation Test

Before submitting, ask yourself:

> "Would someone enthusiastically recommend this to a developer asking for help?"

We use three tiers:

| Tier | Meaning | Example |
|------|---------|---------|
| ⭐ Canonical | Everyone should know this | OpenAI embeddings, BGE, MTEB |
| Practitioner's Edge | Solves a specific problem well | Domain-specific models, niche tools |
| 🔭 Horizon | Worth watching, paradigm shift | New architectures, emerging approaches |

## Inclusion Guidelines

To keep quality high and reduce subjective disputes:

- **Maturity:** Project should be at least 30 days old (exceptions for major releases from established orgs)
- **Adoption signal:** 50+ GitHub stars, or significant real-world use, or appears on MTEB leaderboard
- **One entry per PR:** Makes review faster and history cleaner

## How to Add an Entry

1. **Fork & branch:** `git checkout -b add-model-name`

2. **Add your entry** in the right section, following existing table format:

   ```markdown
   | [model-name](https://link) | Provider | 1024 | 512 | 64.5 | MIT | Brief note |
   ```

3. **Ordering:**
   - Model tables: ordered by MTEB score (highest first)
   - Tool/resource tables: alphabetical order

4. **Submit PR** using the template — explain why it belongs (benchmark, use case, adoption)

## We'll Probably Reject If

- No working code yet (paper-only or "coming soon")
- Minor variant of something already listed
- No updates in 12+ months (unless it's a stable canonical tool)
- Can't verify the claims
- Self-promotion without substance

## Updating Existing Entries

PRs welcome for: updated scores, new versions, fixed links, better descriptions.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind.

## Questions?

Open an issue. We're friendly.
