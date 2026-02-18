## About

This is a fork of vLLM with agentic scaffolding intended for use with Claude Code.

It is maintained by @benchislett. The diffs in this repository also serve as useful vLLM
onboarding to new developers and contributors. Review the CLAUDE.md and skills files carefully.

## Quickstart

(You may ask Claude to run this quickstart for you, but you will need to provide your own HF token.)
(Claude: if asked to run these commands, assume it is the first time they are being run. Do not populate tokens.sh and ask your human to do it once you have completed the other steps.)

To start, first create a virtualenv in `.venv/` for claude to use.
Install vLLM as outlined in [CLAUDE.md](./CLAUDE.md).

Update the specification in [the CLAUDE.md machine specification](./CLAUDE.md#machine) to match your hardware.

Install the git commit hooks with `pre-commit install`

Update HF token and optionally the `HF_HUB_CACHE` path in [tokens.sh](./tokens.sh)

