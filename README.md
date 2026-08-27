# font-ml

Run small local models over font sources.

Not tied to any editor. It reads and writes UFO, takes a model as a
directory of files on disk, and runs it. A person drives it from a
command line, an editor embeds it, an agent calls it as a library or
shells out and reads JSON back. All three get the same behaviour.

Nothing is downloaded and nothing phones home. You point at a folder.

## Not only glyphs

A font is more than a pile of outlines, and the interesting models are
not all glyph models. Kerning is a property of a *pair*. Spacing is a
property of a glyph in company. Feature code is a property of the whole
font. The crate names those jobs separately, so a caller can ask what a
given model actually does rather than assume:

```sh
font-ml tasks
bolden     not built yet  needs: source, glyph, from-master, to-master
complete   not built yet  needs: source, glyph, prefix
generate   not built yet  needs: glyph
spacing    not built yet  needs: source, glyph
kerning    not built yet  needs: source, left, right
field      not built yet  needs: glyph
```

Naming a task is not claiming it works. `implemented` says whether it
does, and today none of them do: the tokenizer, the weight loading and
the forward pass are in place, and the operations built on them are
not.

## For agents

The command line is meant to be driven by a program as readily as by a
person.

- `--json` on every command, one object out.
- **Capability discovery**: `font-ml describe <model> --json` reports
  the model's shape and its tasks with an `implemented` flag and the
  inputs each one needs. A caller finds out what is possible instead of
  guessing and parsing an error message.
- **Stable exit codes**: `0` fine, `2` the request did not make sense,
  `3` it made sense and is not built yet, `4` it failed. "Not built
  yet" and "you asked wrongly" are different answers and should not
  need string matching to tell apart.
- No prompts, no terminal required, no network.

## Two kinds of model

Font models come in two shapes, and both belong here because they
answer different questions.

**Outline models** predict drawing commands: move, line, curve, close,
with coordinates on a grid. The output *is* an outline, so it can go
into a source and be edited. A prediction can be constrained to move
existing points rather than invent new ones, which is what keeps a
master point-compatible with its siblings, and point compatibility is
what interpolation requires.

**Field models** predict a signed distance field: a grid of distances
to the nearest edge. Errors degrade gracefully rather than breaking an
outline, which suits shapes with no fixed point structure to preserve,
such as the stacked, nonlinear composition of Nasta'liq. The output has
to be traced before it is editable, and tracing does not preserve point
structure.

Neither replaces the other. Pick by whether the task has an outline
structure that has to survive: boldening does, generating a Nasta'liq
ligature does not.

## Models are directories

```
my-model/
  config.json          kind, dims, layers, heads, vocab_size, max_len
  weights.safetensors
  vocab.txt            glyph names, then #U XXXX lines (outline models)
```

`config.json` with no `kind` is read as an outline model, so
checkpoints written before this crate existed load unchanged.

## Status

Early. What works today:

- reading a model directory without loading the weights, which is the
  cheap call a model picker makes over every folder it finds
- the outline tokenizer: outlines to token sequences and back, ported
  from the training lab and tested against its id layout
- loading and running an outline model through
  [candle](https://github.com/huggingface/candle)
- `models`, `describe` and `tasks` on the command line

Not yet: any of the tasks, reading UFO sources, and field models, which
are declared in the format so it does not have to change later.

## Rust version

candle **0.9**, not 0.11: 0.11 uses the unstable `stdarch_neon_f16`
feature on aarch64 and needs nightly. Everything here builds on stable,
which is what the editors embedding this pin.

## Testing against a real checkpoint

The unit tests need nothing. To exercise a trained model:

```sh
FONT_ML_TEST_MODEL=/path/to/run cargo test --test real_checkpoint -- --nocapture
```

It checks that the vocabulary this crate builds is the size the
checkpoint was trained with, that the weights load into the right
shapes, and that the logits are finite.

## License

Apache-2.0 OR MIT.
