//! The `piano-tuner` binary's own drivers.
//!
//! Everything here is a **subcommand**: an operational tool that is run again
//! whenever the instrument moves — the preset factory (`fit`, `sympathetic`,
//! `tail`, `noise`, `mics`, alongside `survey`, which is old enough to still live in
//! `main.rs`), the standing boards that write a document into `renders/`
//! (`bench`, `compass`, `melody`, `chain`, `stereo`), and the audits that print
//! (`score`, `brilliance`, `residuals`) or render (`ab`).
//!
//! They are modules of the **binary** target and not of the library, which is
//! what lets them depend on the engine, on a thread pool and on a PNG encoder
//! while `piano_tuner` the library keeps analysing recordings with none of
//! those on its link line (`Cargo.toml`'s `tools` feature).
//!
//! The one-shot instruments that were run once, printed a table into a
//! `DECISIONS.md` item and are kept as that item's reproducibility record are
//! not here: they are `forensics/`, excluded from the workspace's
//! default-members and built on demand.

pub mod ab;
pub mod adapt;
pub mod bench;
pub mod brilliance;
pub mod chain;
pub mod compass;
pub mod fit;
pub mod level;
pub mod listen;
pub mod melody;
pub mod mics;
pub mod noise;
pub mod radiation;
pub mod residuals;
pub mod score;
pub mod stereo;
pub mod sympathetic;
pub mod tail;
