//! Command line shared by every Rust adapter:
//!
//! ```text
//! <adapter> --scene <id>.scene.json --out result.json [--poses poses.bin]
//!           [--mode batch|sync] [--threads N] [--sleep on|off] [--iterations N]
//!           [--set key=value]... [--max-seconds S] [--label text] [--quiet] [--smoke]
//! ```
//! `--smoke` shrinks the protocol to a handful of steps (lead 2, one batch of 2): it proves the
//! adapter builds, loads the scene identically, steps and reads back, using almost no compute.
//! Its timings and gates mean nothing.
//! `--iterations` overrides the engine's own iteration knob (default: the engine's default,
//! or the scene's reference count for AVBD engines). `--threads 0` means "all logical cores".

use crate::protocol::{Mode, RunOptions};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Args {
    pub scene: PathBuf,
    pub out: PathBuf,
    pub threads: Option<usize>,
    pub sleep: bool,
    pub iterations: Option<u32>,
    pub set: BTreeMap<String, String>,
    pub smoke: bool,
    pub run: RunOptions,
}

impl Args {
    pub fn parse() -> Result<Self> {
        Self::parse_from(std::env::args().skip(1))
    }

    pub fn parse_from(it: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut it = it.into_iter();
        let (mut scene, mut out, mut poses) = (None, None, None);
        let (mut threads, mut sleep, mut iterations, mut set) = (None, false, None, BTreeMap::new());
        let (mut mode, mut max_seconds, mut label, mut quiet, mut smoke) = (Mode::Batch, None, None, false, false);
        while let Some(a) = it.next() {
            let mut val = || it.next().with_context(|| format!("{a} needs a value"));
            match a.as_str() {
                "--scene" => scene = Some(PathBuf::from(val()?)),
                "--out" => out = Some(PathBuf::from(val()?)),
                "--poses" => poses = Some(PathBuf::from(val()?)),
                "--mode" => {
                    mode = match val()?.as_str() {
                        "batch" => Mode::Batch,
                        "sync" => Mode::Sync,
                        m => bail!("--mode {m}: expected batch or sync"),
                    }
                }
                "--threads" => threads = Some(val()?.parse().context("--threads")?),
                "--sleep" => {
                    sleep = match val()?.as_str() {
                        "on" => true,
                        "off" => false,
                        s => bail!("--sleep {s}: expected on or off"),
                    }
                }
                "--iterations" => iterations = Some(val()?.parse().context("--iterations")?),
                "--set" => {
                    let kv = val()?;
                    let (k, v) = kv.split_once('=').with_context(|| format!("--set {kv}: expected key=value"))?;
                    set.insert(k.to_string(), v.to_string());
                }
                "--max-seconds" => max_seconds = Some(val()?.parse().context("--max-seconds")?),
                "--label" => label = Some(val()?),
                "--quiet" => quiet = true,
                "--smoke" => smoke = true,
                "-h" | "--help" => {
                    eprintln!("{}", include_str!("cli.rs").lines().take_while(|l| l.starts_with("//!")).map(|l| l.trim_start_matches("//!")).collect::<Vec<_>>().join("\n"));
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        let scene = scene.context("--scene is required")?;
        let out = out.context("--out is required")?;
        let poses = Some(poses.unwrap_or_else(|| out.with_extension("poses.bin")));
        Ok(Self { scene, out, threads, sleep, iterations, set, smoke, run: RunOptions { mode, poses, max_seconds, label, quiet } })
    }

    /// Worker threads for a CPU engine: `--threads N`, 0 or absent meaning every logical core.
    pub fn cpu_threads(&self) -> usize {
        match self.threads {
            Some(n) if n > 0 => n,
            _ => crate::machine::logical_cores(),
        }
    }

    /// Take an engine-specific `--set` value, so unknown keys can be rejected afterwards.
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.set.remove(key)
    }

    pub fn reject_unknown(&self) -> Result<()> {
        if !self.set.is_empty() {
            bail!("unknown --set keys for this adapter: {:?}", self.set.keys().collect::<Vec<_>>());
        }
        Ok(())
    }
}
