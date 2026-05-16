# Contributing translations

rlox uses [`mkdocs-static-i18n`](https://github.com/ultrabug/mkdocs-static-i18n)
to serve the documentation in multiple languages from a single source tree.
Translating a page is a pure markdown edit — no tooling changes needed.

## Quick recipe

1. Pick a page under `docs/` that you want to translate, e.g. `docs/examples.md`.
2. Copy it to `docs/examples.<lang>.md`, where `<lang>` is a short locale code:
   - `pl` — Polish (Polski)
   - `de` — German (Deutsch)
   - `fr` — French (Français)
3. Translate the prose. **Do not translate code, identifiers, file paths,
   environment names, or CLI flags.** Those are part of the API and must
   stay byte-identical.
4. Build the site locally to check:
   ```bash
   ./.venv/bin/python -m mkdocs build
   ./.venv/bin/python -m mkdocs serve  # live preview at http://127.0.0.1:8000
   ```
5. Open a PR. Tag `@wojciechkpl` and include a one-line summary of the page
   and target language.

That's it. No build-system changes, no CI edits, no YAML reconfiguration —
`mkdocs-static-i18n` picks up the new file automatically.

## What's currently translated

| page | en | pl | de | fr |
|---|:-:|:-:|:-:|:-:|
| `index.md` (landing) | ✅ | ✅ | ✅ | ✅ |
| `getting-started.md` | ✅ | 🟡 abridged | 🟡 abridged | 🟡 abridged |
| `learning-path.md` | ✅ | 🟡 abridged | 🟡 abridged | 🟡 abridged |
| everything else | ✅ | ⏸ falls back to en | ⏸ falls back to en | ⏸ falls back to en |

Pages marked **abridged** cover only the first section or two plus a
pointer to the full English version. Extending them to the full content
is welcome — see the "Extending an abridged page" section below.

Pages marked **⏸** have no translation file yet; clicking a link to them
from a translated page will serve the English content under the translated
URL (e.g. `/pl/api/index/`). This is the built-in fallback behavior of
`mkdocs-static-i18n` with `fallback_to_default: true`.

## Adding a new language

To add a locale we don't already support (say, Spanish = `es`):

1. Edit `mkdocs.yml` and add an entry under `plugins.i18n.languages`:
   ```yaml
   - locale: es
     name: Español
     build: true
     site_name: "rlox — Aprendizaje por refuerzo acelerado por Rust"
   ```
2. Translate `docs/index.es.md` (at minimum the landing page, so the
   language switcher has something to route to).
3. Build + verify locally.
4. Open a PR with the `mkdocs.yml` change + the new `.es.md` file.

## Style guide

### What to translate

- Section headings, body prose, image `alt` text.
- The `title:` front-matter field if present.
- Marketing copy in the landing page.
- User-facing notes in admonitions (e.g. "Tip:", "Warning:").

### What NOT to translate

- **Code blocks**, inline code, variable names, function names, file
  paths, URLs, environment IDs (`CartPole-v1`, `Hopper-v4`), CLI flags
  (`--timesteps`), commit hashes, package names (`maturin`, `gymnasium`).
- **Mermaid diagram node IDs** — they are identifiers, not display text.
  You *can* translate the label inside `[...]` brackets, e.g.
  `L1[Level 1: Getting Started]` → `L1[Poziom 1: pierwsze kroki]`.
- **The file names** of the original English pages when you link to
  them. Links within translated pages can use normal relative paths and
  `mkdocs-static-i18n` resolves them to the correct locale.
- **Algorithm acronyms** — PPO, SAC, TD3, DQN, GAE, GRPO, DPO, KL, etc.
  These are universal in the RL literature. Expand the full name only
  on first use if your target language has a conventional expansion.

### Tone

- Keep the pragmatic "this is what you type, this is what you get" style
  of the English docs. Don't inflate into academic prose.
- Use the formal form where your language distinguishes (e.g. German
  *Sie*, French *vous*). rlox targets researchers and engineers who
  appreciate precision over familiarity.
- Preserve the English admonition labels' intent but use the target-
  language word (e.g. "Tip" → "Wskazówka" / "Tipp" / "Astuce").

## File naming convention

| rule | example |
|---|---|
| Default English file | `docs/index.md` |
| Translated variant | `docs/index.<lang>.md` |
| Nested docs | `docs/tutorials/custom-components.pl.md` |
| Locale code | lowercase two-letter ISO 639-1 (`pl`, `de`, `fr`, `es`, `ja`, ...) |

The plugin is configured with `docs_structure: suffix` so the `.<lang>`
suffix on the filename is what matters — directory structure stays the
same as the English tree.

## Extending an abridged page

If you want to turn an abridged translation into a full one:

1. Read the English source (`docs/<page>.md`) top to bottom.
2. Open the abridged translation (`docs/<page>.<lang>.md`).
3. Delete the abridged-translation banner at the top.
4. Replace the "Further reading" / "Next steps" section with the
   translations of the remaining English sections.
5. Keep all code blocks byte-identical to the English source.
6. Build locally and open a PR.

Rough time estimate: a fully-translated `getting-started.md` takes
~2 hours for a native speaker who also codes in Python.

## Translating the blog

The blog lives in a separate tree (`blog/` via Hugo) and has its own
i18n configuration. The recipe above does NOT apply to blog posts —
see the Hugo `content/<lang>/...` convention instead. If you want to
translate a blog post, open an issue first so we can discuss scope.

## Review process

- Translations are reviewed by at least one native speaker of the target
  language before merging.
- For algorithmic correctness we also need one reviewer with a working
  understanding of RL, since a linguistically-correct translation that
  misdescribes GAE or PPO is a net negative.
- Machine-translated PRs are politely closed with a link to this guide.
  We only merge translations that a human has read and understood.

## Questions

Open a discussion at https://github.com/wojciechkpl/rlox/discussions with
the `translations` label, or ping `@wojciechkpl` on the PR directly.
