//! Running a subprocess inside a container.
//!
//! [`Confinement::PathChecked`] validates the paths a *file* operation names,
//! but once `cargo test` is running nothing stops that process opening an
//! absolute path or a socket. For the Warden — which is trying to pass tests
//! — that is an acceptable trust boundary. For the Deviant, which is
//! explicitly optimised to escape, it is not, and the design says plainly
//! that it must not be run under `PathChecked`.
//!
//! This is the boundary that makes the arms race safe to hold.
//!
//! # What the flags are for
//!
//! Each one closes a specific route out, and none is decorative:
//!
//! | flag | what it stops |
//! |---|---|
//! | `--network none` | exfiltration, and fetching the answer off the internet |
//! | `--read-only` + `tmpfs` | persisting anything outside the episode |
//! | one bind mount | reaching any host path but the sandbox |
//! | `--cap-drop ALL` | everything capabilities would otherwise permit |
//! | `--security-opt no-new-privileges` | setuid escalation |
//! | `--pids-limit` | a fork bomb taking the host down |
//! | `--memory` / `--cpus` | starving the host of resources |
//! | `--rm` | state leaking from one episode into the next |
//!
//! # What it is not
//!
//! A container is an OS-level isolation boundary, not a formal one. A kernel
//! exploit escapes it, and saying otherwise would be the kind of overclaim
//! this codebase is trying to avoid elsewhere. What it is good for is the
//! actual threat model: an agent driving a shell, trying things, and being
//! rewarded for finding a way out. Against that it is a real wall, and the
//! difference between it and `PathChecked` is the difference between a wall
//! and a request.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Which container CLI to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    Docker,
    Nerdctl,
    Podman,
}

impl Runtime {
    pub fn program(self) -> &'static str {
        match self {
            Runtime::Docker => "docker",
            Runtime::Nerdctl => "nerdctl",
            Runtime::Podman => "podman",
        }
    }

    /// The first runtime on `PATH`, in order of how well tested each is here.
    pub fn detect() -> Option<Runtime> {
        [Runtime::Docker, Runtime::Nerdctl, Runtime::Podman]
            .into_iter()
            .find(|r| which(r.program()))
    }
}

fn which(program: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    std::process::Command::new(probe)
        .arg(program)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// How tightly a container is held.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerPolicy {
    pub runtime: Runtime,
    /// Image to run. Must already contain whatever the task's oracle needs;
    /// the container has no network and cannot fetch a toolchain.
    pub image: String,
    /// Where the sandbox appears inside the container.
    pub workdir: String,
    pub memory: String,
    pub cpus: String,
    pub pids_limit: u32,
    /// Size of the writable `/tmp`. Some toolchains cannot run without one,
    /// and a read-only root would otherwise make the image unusable.
    pub tmpfs_size: String,
}

impl ContainerPolicy {
    /// A policy using whatever runtime is installed.
    pub fn detect(image: impl Into<String>) -> Option<Self> {
        Runtime::detect().map(|runtime| Self {
            runtime,
            image: image.into(),
            workdir: "/work".into(),
            memory: "2g".into(),
            cpus: "2".into(),
            pids_limit: 512,
            tmpfs_size: "64m".into(),
        })
    }

    pub fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = runtime;
        self
    }

    /// Build the full argv for running `program` with `args` against `sandbox`.
    ///
    /// Pure, so the flags can be tested without a runtime installed — which
    /// matters, because the flags *are* the security property and a test that
    /// only runs where Docker happens to exist is a test that silently
    /// disappears on the machine you most wanted it on.
    pub fn argv(&self, sandbox: &Path, program: &str, args: &[String]) -> Vec<String> {
        let mut v: Vec<String> = vec![
            "run".into(),
            "--rm".into(),
            // No route off the machine, and no route to the answer.
            "--network".into(),
            "none".into(),
            // Nothing survives outside the mounted sandbox.
            "--read-only".into(),
            "--tmpfs".into(),
            format!("/tmp:rw,noexec,nosuid,size={}", self.tmpfs_size),
            // Exactly one host path is visible.
            "--mount".into(),
            format!(
                "type=bind,src={},dst={}",
                mount_source(sandbox),
                self.workdir
            ),
            "--workdir".into(),
            self.workdir.clone(),
            // Nothing to escalate with.
            "--cap-drop".into(),
            "ALL".into(),
            "--security-opt".into(),
            "no-new-privileges".into(),
            // A runaway cannot take the host with it.
            "--memory".into(),
            self.memory.clone(),
            "--cpus".into(),
            self.cpus.clone(),
            "--pids-limit".into(),
            self.pids_limit.to_string(),
            self.image.clone(),
            program.to_string(),
        ];
        v.extend(args.iter().cloned());
        v
    }
}

/// Render a host path as the runtime expects it for a bind mount.
///
/// Windows paths need converting for a Linux container: `C:\a\b` becomes
/// `/c/a/b` wherever the runtime is fronting a WSL or Lima VM, which covers
/// Docker Desktop, Rancher Desktop and podman machine. A verbatim `\\?\`
/// prefix — which `canonicalize` produces on Windows, and the sandbox path is
/// always canonicalised — is stripped first, because no runtime understands
/// it.
pub fn mount_source(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    if !cfg!(windows) {
        return s;
    }

    let s = s
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .unwrap_or_else(|| s.strip_prefix(r"\\?\").unwrap_or(&s).to_string());

    let bytes: Vec<char> = s.chars().collect();
    if bytes.len() >= 2 && bytes[1] == ':' && bytes[0].is_ascii_alphabetic() {
        let drive = bytes[0].to_ascii_lowercase();
        let rest: String = bytes[2..].iter().collect::<String>().replace('\\', "/");
        return format!("/{drive}{rest}");
    }
    s.replace('\\', "/")
}
