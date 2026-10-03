# AI Disclosure

Team SRMIST_NOVA, Samsung PRISM GenAI Hackathon 2026, Theme 1.

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Anthropic: Claude Opus 5, Claude Opus 5.5, Claude Sonnet 5) | Coding assistant: writing and reviewing code, tests, Docker setup, documentation, the command-line demo (`scripts/search.py`) and the demo video recording |

## How it was used

- **Code and docs.** Claude Code wrote or edited code and documentation under
  the team's direction. Commits where it contributed carry a
  `Co-Authored-By: Claude ...` trailer, so the git history shows exactly which
  changes it was involved in (9 of the 43 commits at the time of writing).
- **Measurements and experiments.** Every number in `experiments.md`,
  `README.md` and the presentation comes from scripts that were actually run
  on this repository. The scripts and their output files are in `scripts/`
  and `data/`.
- **Demo video.** `docs/demo.mp4` is a recording of real runs of
  `scripts/search.py` inside the Docker image with `--network none`. The
  terminal output and the timings shown are what the program printed. Typing
  is animated, idle waits longer than 2.5 s are shortened, and there are short
  pauses between results so they can be read.

## What the team decided

The team chose the problem approach, the model and the evaluation setup, and
reviewed and accepted every change. AI did not decide what to ship. Choices
such as keeping BM25 fusion and reranking off were made by the team from
measured results, recorded in `experiments.md`.

## The retrieval system itself

The submitted pipeline does not call any generative AI model or external API.
It is a dense bi-encoder (`Snowflake/snowflake-arctic-embed-m`, open weights)
that runs offline on CPU.
