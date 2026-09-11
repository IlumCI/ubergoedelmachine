//! A thin wrapper over the `git` CLI.
//!
//! Deliberately the CLI rather than `libgit2`. Half of what this crate needs
//! is porcelain that libgit2 does not expose anyway (`worktree add`, shallow
//! `fetch`), the output formats used here are stable and machine-oriented, and
//! it removes a vendored C dependency from a build that has already proven
//! delicate on this toolchain.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

use crate::CorpusError;

/// Run git in `dir` and return stdout, trimmed of the trailing newline.
pub fn git<I, S>(dir: &Path, args: I) -> Result<String, CorpusError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let argv: Vec<String> = args
        .into_iter()
        .map(|a| a.as_ref().to_string_lossy().into_owned())
        .collect();

    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        // Keep the caller's git config from changing what we parse.
        .args(["-c", "core.quotepath=false"])
        .args(&argv)
        .output()
        .map_err(|e| CorpusError::GitSpawn {
            detail: e.to_string(),
        })?;

    if !out.status.success() {
        return Err(CorpusError::Git {
            args: argv.join(" "),
            status: out.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .trim_end_matches(['\n', '\r'])
        .to_string())
}

/// Run git and return the raw bytes of stdout, for file contents that are not
/// necessarily UTF-8.
pub fn git_bytes<I, S>(dir: &Path, args: I) -> Result<Vec<u8>, CorpusError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let argv: Vec<String> = args
        .into_iter()
        .map(|a| a.as_ref().to_string_lossy().into_owned())
        .collect();

    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(&argv)
        .output()
        .map_err(|e| CorpusError::GitSpawn {
            detail: e.to_string(),
        })?;

    if !out.status.success() {
        return Err(CorpusError::Git {
            args: argv.join(" "),
            status: out.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok(out.stdout)
}

/// Whether git can see a commit object. Used to prove a sandbox *cannot*.
pub fn has_commit(dir: &Path, sha: &str) -> bool {
    git(dir, ["cat-file", "-e", &format!("{sha}^{{commit}}")]).is_ok()
}
