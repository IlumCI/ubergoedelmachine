//! Running subprocesses with a deadline.
//!
//! `std::process` has no timeout, and the naive fix — wait, then `kill()` —
//! is not enough for anything the harness actually runs. `cargo test` spawns
//! `rustc` and then the test binaries; killing `cargo` leaves those running,
//! holding file handles in the worktree that is about to be deleted, and on
//! Windows a held handle makes removal fail rather than merely being untidy.
//!
//! So termination goes after the whole process tree: `taskkill /T` on Windows,
//! a process-group kill on Unix.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// How a subprocess ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Completion {
    Exited { code: i32 },
    /// Killed for exceeding its deadline.
    TimedOut { after_secs: u64 },
    /// Could not be started: missing program, bad working directory.
    Unstartable { detail: String },
}

impl Completion {
    pub fn success(&self) -> bool {
        matches!(self, Completion::Exited { code: 0 })
    }
}

/// What running a command produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub completion: Completion,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_secs: f64,
    /// Whether output was cut off at the cap.
    pub truncated: bool,
}

impl Output {
    pub fn success(&self) -> bool {
        self.completion.success()
    }
}

/// Cap on captured output.
///
/// A runaway process can emit gigabytes. Since this text ends up as evidence
/// in the ledger, and the ledger is meant to be readable, it is bounded here
/// rather than at the point where it has already been written to disk.
const MAX_CAPTURE: usize = 256 * 1024;

fn truncate(mut s: String) -> (String, bool) {
    if s.len() <= MAX_CAPTURE {
        return (s, false);
    }
    s.truncate(MAX_CAPTURE);
    s.push_str("\n[... truncated ...]");
    (s, true)
}

/// Run `program` with `args` in `cwd`, killing it after `timeout`.
pub fn run(
    program: &str,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    env: &[(String, String)],
) -> Output {
    let started = Instant::now();

    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (k, v) in env {
        cmd.env(k, v);
    }

    // On Unix, put the child in its own process group so the whole group can
    // be signalled. Windows gets the same effect from `taskkill /T`.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc_setsid();
                Ok(())
            });
        }
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return Output {
                completion: Completion::Unstartable {
                    detail: e.to_string(),
                },
                stdout: String::new(),
                stderr: String::new(),
                elapsed_secs: started.elapsed().as_secs_f64(),
                truncated: false,
            };
        }
    };

    // Drain the pipes on threads. Without this a child that fills a pipe
    // buffer blocks forever and the timeout fires on a process that was only
    // ever waiting for us to read it.
    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(p) = out_pipe.as_mut() {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(p) = err_pipe.as_mut() {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });

    let pid = child.id();
    let completion;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                completion = Completion::Exited {
                    code: status.code().unwrap_or(-1),
                };
                break;
            }
            Ok(None) => {}
            Err(e) => {
                completion = Completion::Unstartable {
                    detail: e.to_string(),
                };
                break;
            }
        }

        if started.elapsed() >= timeout {
            kill_tree(pid);
            let _ = child.kill();
            let _ = child.wait();
            completion = Completion::TimedOut {
                after_secs: timeout.as_secs(),
            };
            break;
        }

        std::thread::sleep(Duration::from_millis(25));
    }

    let stdout = String::from_utf8_lossy(&out_handle.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err_handle.join().unwrap_or_default()).into_owned();
    let (stdout, t1) = truncate(stdout);
    let (stderr, t2) = truncate(stderr);

    Output {
        completion,
        stdout,
        stderr,
        elapsed_secs: started.elapsed().as_secs_f64(),
        truncated: t1 || t2,
    }
}

/// Kill a process and everything it spawned.
#[cfg(windows)]
fn kill_tree(pid: u32) {
    // /T takes the tree, /F does not ask nicely. Failure is ignored: the
    // process may have exited between the timeout check and here.
    let _ = Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn kill_tree(pid: u32) {
    // Negative pid signals the process group established by setsid above.
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn libc_setsid() {
    unsafe extern "C" {
        fn setsid() -> i32;
    }
    unsafe {
        setsid();
    }
}

/// What [`normalize_command`] did, if anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Normalization {
    /// The `program` string as the caller supplied it.
    pub from: String,
    /// What it was split into.
    pub program: String,
    pub args: Vec<String>,
}

/// Split a fused command string into program and arguments.
///
/// Models routinely emit `{"program": "cargo build", "args": []}`, which
/// spawns a binary named `cargo build` and fails with a confusing "program
/// not found". Fixing that in the prompt is possible but fragile — it is one
/// more instruction competing for attention in a model whose
/// instruction-following was already damaged by ablation, and it fails
/// silently when it does not take. Accepting the input here is robust, and
/// costs nothing when the caller got it right.
///
/// Deliberately conservative, because a Windows executable path legitimately
/// contains spaces and splitting `C:\Program Files\Git\bin\git.exe` would
/// break something that was correct. Four conditions must all hold:
///
/// 1. `args` is empty — a caller that supplied arguments understood the
///    schema, and the program string is theirs to keep.
/// 2. `program` contains whitespace outside of quotes.
/// 3. `program` is not itself an existing file, so real paths with spaces
///    survive untouched.
/// 4. The split yields a non-empty program.
///
/// The result is reported rather than applied silently: the ledger should
/// show what was actually run next to what was asked for, because an
/// execution that quietly differs from the request is the kind of thing this
/// harness exists to make visible.
pub fn normalize_command(program: &str, args: &[String]) -> Option<Normalization> {
    if !args.is_empty() {
        return None;
    }
    let trimmed = program.trim();
    if trimmed.is_empty() {
        return None;
    }

    let parts = split_command(trimmed);
    if parts.len() < 2 {
        return None;
    }

    // A real path that happens to contain spaces. Only reachable when the
    // file exists, so this cannot be used to smuggle anything: a nonexistent
    // path still splits, and a split command still goes through the jail and
    // the router like any other.
    if std::path::Path::new(trimmed).is_file() {
        return None;
    }

    let (head, tail) = parts.split_first()?;
    if head.is_empty() {
        return None;
    }

    Some(Normalization {
        from: program.to_string(),
        program: head.clone(),
        args: tail.to_vec(),
    })
}

/// Whitespace split that respects double quotes.
///
/// Not a full shell parser, and deliberately not one — the executor does not
/// run a shell, so honouring shell metacharacters here would imply a
/// capability that does not exist. Quotes are handled only because a quoted
/// path is the common way to write an executable containing spaces.
fn split_command(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;

    for c in s.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            c if c.is_whitespace() && !in_quotes => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}
