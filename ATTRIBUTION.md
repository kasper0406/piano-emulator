# Attribution

The measured preset `presets/salamander-c5.toml` — its tuning, inharmonicity,
decay, unison detuning, excitation texture, micro-motion, stereo directivity
and action-noise tables — was estimated from recordings of a Yamaha C5
published as the **Salamander Grand Piano** (V3) by **Alexander Holm**,
licensed **CC-BY 3.0**: <https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html>.

The recordings themselves are not distributed with this repository; the
gitignored `data/fetch_salamander.sh` downloads them (checksummed) for anyone
re-running the estimation pipeline, and the reference side of the realism
benchmark plays them back for comparison.

Action-noise levels and structure-borne bandwidth were cross-checked against
figures published in the piano-acoustics literature (Askenfelt; Goebl, Bresin
& Galembo; Lehtonen, Askenfelt & Välimäki), cited where used in `PHYSICS.md`
and `DECISIONS.md`.

Any future sample library used by the estimation pipeline must have its
license recorded here and in its fetch script before its parameters ship in a
preset.
