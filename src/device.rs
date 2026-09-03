//! Which device a model runs on.
//!
//! One choice for every command that runs a model: `auto`, `cpu`,
//! `metal`, or `cuda[:n]`. Auto takes the GPU the binary was built
//! for when one is there, else the CPU, so an installed binary does
//! the right thing with no flag and a script can still pin a device.

use candle_core::Device;

use crate::error::{Error, Result};

/// What was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Choice {
    /// A GPU the build supports if one answers, else the CPU.
    #[default]
    Auto,
    /// The CPU, whatever was built.
    Cpu,
    /// Apple's GPU. Needs the `metal` feature.
    Metal,
    /// NVIDIA, by ordinal. Needs the `cuda` feature.
    Cuda(usize),
}

impl std::str::FromStr for Choice {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "auto" => Ok(Choice::Auto),
            "cpu" => Ok(Choice::Cpu),
            "metal" => Ok(Choice::Metal),
            "cuda" => Ok(Choice::Cuda(0)),
            other => match other.strip_prefix("cuda:") {
                Some(n) => n
                    .parse()
                    .map(Choice::Cuda)
                    .map_err(|_| format!("bad CUDA ordinal in {other}")),
                None => Err(format!(
                    "unknown device {other}; use auto, cpu, metal, or cuda[:n]"
                )),
            },
        }
    }
}

impl std::fmt::Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Auto => f.write_str("auto"),
            Choice::Cpu => f.write_str("cpu"),
            Choice::Metal => f.write_str("metal"),
            Choice::Cuda(n) => write!(f, "cuda:{n}"),
        }
    }
}

fn unavailable(what: &str, feature: &str) -> Error {
    Error::Io {
        path: std::path::PathBuf::from(what),
        source: std::io::Error::other(format!(
            "{what} is not available: build font-ml with --features {feature}, and \
             check that a device answers"
        )),
    }
}

/// The device for a choice. `Auto` never fails; a named GPU that is
/// not built in or does not answer is an error, so a script that
/// pinned a device is told rather than silently run on the CPU.
pub fn pick(choice: Choice) -> Result<Device> {
    match choice {
        Choice::Cpu => Ok(Device::Cpu),
        Choice::Metal => metal(0).ok_or_else(|| unavailable("metal", "metal")),
        Choice::Cuda(n) => cuda(n).ok_or_else(|| unavailable("cuda", "cuda")),
        Choice::Auto => Ok(metal(0).or_else(|| cuda(0)).unwrap_or(Device::Cpu)),
    }
}

#[cfg(feature = "metal")]
fn metal(ordinal: usize) -> Option<Device> {
    Device::new_metal(ordinal).ok()
}

#[cfg(not(feature = "metal"))]
fn metal(_ordinal: usize) -> Option<Device> {
    None
}

#[cfg(feature = "cuda")]
fn cuda(ordinal: usize) -> Option<Device> {
    Device::new_cuda(ordinal).ok()
}

#[cfg(not(feature = "cuda"))]
fn cuda(_ordinal: usize) -> Option<Device> {
    None
}

/// The device's name for a report: `cpu`, `metal`, or `cuda`.
pub fn name(device: &Device) -> &'static str {
    if device.is_metal() {
        "metal"
    } else if device.is_cuda() {
        "cuda"
    } else {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_parse_and_print() {
        assert_eq!("auto".parse::<Choice>().unwrap(), Choice::Auto);
        assert_eq!("CPU".parse::<Choice>().unwrap(), Choice::Cpu);
        assert_eq!("cuda".parse::<Choice>().unwrap(), Choice::Cuda(0));
        assert_eq!("cuda:1".parse::<Choice>().unwrap(), Choice::Cuda(1));
        assert!("tpu".parse::<Choice>().is_err());
        assert_eq!(Choice::Cuda(1).to_string(), "cuda:1");
    }

    #[test]
    fn cpu_and_auto_always_answer() {
        assert!(pick(Choice::Cpu).unwrap().is_cpu());
        let d = pick(Choice::Auto).unwrap();
        assert!(["cpu", "metal", "cuda"].contains(&name(&d)));
    }
}
