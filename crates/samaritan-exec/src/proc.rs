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
