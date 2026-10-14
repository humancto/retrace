## What this pull request changes

<!-- One paragraph. Link the issue if there is one: Closes #123 -->

## Why

<!-- The problem that this solves, from the side of the user. -->

## How you tested it

- [ ] `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings -A clippy::too_many_arguments -A clippy::large_enum_variant` report no finding
- [ ] `cargo test` passes
- [ ] `scripts/e2e.sh` passes
- [ ] For a search or tracker change: `retrace eval --golden eval/golden.vtest.json` before and after

```
# paste the eval line(s) here if relevant
```

## Checklist

- [ ] The change works with no network connection (no new cloud call)
- [ ] Each new model has a permissive license and a download that each person can repeat
- [ ] README and CHANGELOG are current if the behaviour changed
