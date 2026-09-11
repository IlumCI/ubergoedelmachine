//! Keeping file operations inside the episode sandbox.
//!
//! Every path an agent supplies is hostile input. The Deviant is explicitly
//! rewarded for reaching outside its worktree, and this module is the thing
//! standing in the way, so it is written to reject rather than to repair: a
//! path that is unusual in any way is refused, not normalised into something
//! that looks safe.
//!
//! Windows supplies a long list of ways to name a file that does not look like
//! the file it names — drive-relative paths, UNC and verbatim prefixes,
//! alternate data streams, reserved device names, short 8.3 aliases. Combined
//! with symlinks and `..`, prefix-matching a joined path string is not close
//! to sufficient. The approach here is:
//!
//! 1. Reject structurally: absolute, rooted, prefixed, or `..`-bearing paths,
//!    and anything naming a reserved device or a data stream.
//! 2. Resolve physically: canonicalise the deepest ancestor that exists, which
//!    is what collapses symlinks, then confirm the result is still under the
//!    canonicalised root.
//!
//! Step 2 is the one that matters. Step 1 alone can be defeated by a symlink
//! the agent created on a previous action.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Why a path was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathRefusal {
    Empty,
    /// An absolute path, a rooted one, or one carrying a drive or UNC prefix.
    NotRelative { path: String },
    /// Contains a `..` component.
    Traversal { path: String },
    /// Names a Windows reserved device such as `CON` or `NUL`, which resolve
    /// to devices regardless of the directory they appear in.
    ReservedName { segment: String },
    /// Contains a colon, which on NTFS opens an alternate data stream.
    DataStream { segment: String },
    /// Resolved, via symlinks or otherwise, to somewhere outside the sandbox.
    Escapes { path: String, resolved: String },
    /// The path could not be resolved well enough to judge it.
    Unresolvable { path: String, detail: String },
}

impl std::fmt::Display for PathRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathRefusal::Empty => write!(f, "empty path"),
            PathRefusal::NotRelative { path } => {
                write!(f, "{path} is not a relative path")
            }
            PathRefusal::Traversal { path } => write!(f, "{path} contains .."),
            PathRefusal::ReservedName { segment } => {
                write!(f, "{segment} is a reserved device name")
            }
            PathRefusal::DataStream { segment } => {
                write!(f, "{segment} names an alternate data stream")
            }
            PathRefusal::Escapes { path, resolved } => {
                write!(f, "{path} resolves to {resolved}, outside the sandbox")
            }
            PathRefusal::Unresolvable { path, detail } => {
                write!(f, "could not resolve {path}: {detail}")
            }
        }
    }
}

/// Windows device names, which are reserved in every directory. Writing to
/// `logs/NUL` does not create a file called `NUL`.
const RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

fn is_reserved(segment: &str) -> bool {
    // The rule applies to the stem, so `NUL.txt` is still the null device.
    let stem = segment.split('.').next().unwrap_or(segment);
    RESERVED.iter().any(|r| stem.eq_ignore_ascii_case(r))
}

/// A sandbox root, with the ability to resolve agent-supplied paths into it.
#[derive(Debug, Clone)]
pub struct Jail {
    /// Canonicalised, so comparisons are against physical reality rather than
    /// against whatever string the caller passed in.
    root: PathBuf,
}

impl Jail {
    /// Establish a jail at `root`, which must already exist.
    pub fn new(root: &Path) -> Result<Self, PathRefusal> {
        let root = root
            .canonicalize()
            .map_err(|e| PathRefusal::Unresolvable {
                path: root.display().to_string(),
                detail: e.to_string(),
            })?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve an agent-supplied relative path to an absolute one inside the
    /// jail, or refuse it.
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, PathRefusal> {
        if raw.trim().is_empty() {
            return Err(PathRefusal::Empty);
        }

        let normalised = raw.replace('\\', "/");

        // Segment-level checks first, on the string, because some of these are
        // invisible after the path is parsed.
        for segment in normalised.split('/') {
            if segment.is_empty() || segment == "." {
                continue;
            }
            if segment.contains(':') {
                return Err(PathRefusal::DataStream {
                    segment: segment.to_string(),
                });
            }
            if is_reserved(segment) {
                return Err(PathRefusal::ReservedName {
                    segment: segment.to_string(),
                });
            }
        }

        let candidate = Path::new(&normalised);

        // Structural checks. `Prefix` catches `C:` and `\\?\` and UNC shares;
        // `RootDir` catches a leading slash even without a drive.
        for component in candidate.components() {
            match component {
                Component::ParentDir => {
                    return Err(PathRefusal::Traversal {
                        path: raw.to_string(),
                    });
                }
                Component::Prefix(_) | Component::RootDir => {
                    return Err(PathRefusal::NotRelative {
                        path: raw.to_string(),
                    });
                }
                _ => {}
            }
        }
        if candidate.is_absolute() {
            return Err(PathRefusal::NotRelative {
                path: raw.to_string(),
            });
        }

        let joined = self.root.join(candidate);

        // Physical resolution. The target may not exist yet — a write creates
        // it — so canonicalise the deepest ancestor that does and re-attach the
        // rest. This is what catches a symlink the agent planted earlier.
        let resolved = resolve_existing_ancestor(&joined).map_err(|e| {
            PathRefusal::Unresolvable {
                path: raw.to_string(),
                detail: e,
            }
        })?;

        if !resolved.starts_with(&self.root) {
            return Err(PathRefusal::Escapes {
                path: raw.to_string(),
                resolved: resolved.display().to_string(),
            });
        }

        Ok(resolved)
    }
}

/// Canonicalise as much of `path` as exists, then re-append the remainder.
fn resolve_existing_ancestor(path: &Path) -> Result<PathBuf, String> {
    let mut missing = Vec::new();
    let mut cursor = path.to_path_buf();

    loop {
        match cursor.canonicalize() {
            Ok(real) => {
                let mut out = real;
                for part in missing.iter().rev() {
                    out.push(part);
                }
                return Ok(out);
            }
            Err(_) => {
                let Some(name) = cursor.file_name().map(|n| n.to_os_string()) else {
                    return Err("no existing ancestor".to_string());
                };
                missing.push(name);
                if !cursor.pop() {
                    return Err("no existing ancestor".to_string());
                }
            }
        }
    }
}
