## What this changes

<!-- One paragraph. Link the issue if there is one: Closes #123 -->

## Why

<!-- The problem this solves from the user's side of the screen. -->

## How it was tested

- [ ] `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings -A clippy::too_many_arguments -A clippy::large_enum_variant` are clean
- [ ] `cargo test` passes
- [ ] `scripts/e2e.sh` passes
- [ ] For search or tracking changes: `retrace eval --golden eval/golden.vtest.json` before and after

```
# paste the eval line(s) here if relevant
```

## Checklist

- [ ] Works with the network cable pulled (no new cloud calls)
- [ ] Any new model has a permissive license and a reproducible download
- [ ] README / CHANGELOG updated if behaviour changed
