//! Tests for the executor.
//!
//! The confinement cases are written as an attacker would write them. Each one
//! is a real way to name a file outside a directory, and several of them look
//! like ordinary relative paths right up until the filesystem resolves them.

use std::path::Path;

use samaritan_dsl::{ActionKind, BlastRadius, ProposedAction, Reversibility};
use samaritan_exec::confine::{Jail, PathRefusal};
use samaritan_exec::{Confinement, ExecError, Executor, Op, Outcome};

fn jail_dir() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("sandbox");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "fn a() {}\n").unwrap();
    (dir, root)
}

fn exec() -> (tempfile::TempDir, Executor) {
    let (dir, root) = jail_dir();
    let e = Executor::new(&root, Confinement::PathChecked).unwrap();
    (dir, e)
}

fn action(payload: serde_json::Value) -> ProposedAction {
    ProposedAction {
        kind: ActionKind::Write,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Episode,
        intent: "do the thing".into(),
        payload,
    }
}

fn is_escape(o: &Outcome) -> bool {
    o.violations
        .iter()
        .any(|v| v.tag == "sandbox_escape_attempt")
}

// -------------------------------------------------------------- refusals

#[test]
fn ordinary_relative_paths_are_allowed() {
    let (_g, root) = jail_dir();
    let jail = Jail::new(&root).unwrap();
    for ok in ["src/lib.rs", "./src/lib.rs", "new.txt", "a/b/c/deep.txt"] {
        assert!(jail.resolve(ok).is_ok(), "{ok} should resolve");
    }
}

#[test]
fn traversal_is_refused() {
    let (_g, root) = jail_dir();
    let jail = Jail::new(&root).unwrap();
    for bad in ["../outside.txt", "src/../../outside.txt", "a/b/../../.."] {
        assert!(
            matches!(jail.resolve(bad), Err(PathRefusal::Traversal { .. })),
            "{bad} should be refused as traversal"
        );
    }
}

#[test]
fn absolute_and_drive_relative_paths_are_refused() {
    let (_g, root) = jail_dir();
    let jail = Jail::new(&root).unwrap();
    for bad in [
        "/etc/passwd",
        "C:/Windows/System32/drivers/etc/hosts",
        r"C:\Windows\win.ini",
        r"\\server\share\file",
        r"\\?\C:\Windows\win.ini",
    ] {
        let r = jail.resolve(bad);
        assert!(
            r.is_err(),
            "{bad} should be refused, got {:?}",
            r.map(|p| p.display().to_string())
        );
    }
}

#[test]
fn windows_device_names_are_refused_in_any_directory() {
    // `logs/NUL` does not create a file called NUL; it opens the null device.
    let (_g, root) = jail_dir();
    let jail = Jail::new(&root).unwrap();
    for bad in ["NUL", "CON", "src/NUL", "logs/con.txt", "aux", "COM1"] {
        assert!(
            matches!(jail.resolve(bad), Err(PathRefusal::ReservedName { .. })),
            "{bad} should be refused as a device name"
        );
    }
}

#[test]
fn alternate_data_streams_are_refused() {
    // `file.txt:hidden` writes a stream most tooling never shows.
    let (_g, root) = jail_dir();
    let jail = Jail::new(&root).unwrap();
    assert!(matches!(
        jail.resolve("src/lib.rs:hidden"),
        Err(PathRefusal::DataStream { .. })
    ));
}

#[test]
fn an_empty_path_is_refused() {
    let (_g, root) = jail_dir();
    let jail = Jail::new(&root).unwrap();
    assert_eq!(jail.resolve(""), Err(PathRefusal::Empty));
    assert_eq!(jail.resolve("   "), Err(PathRefusal::Empty));
}

#[test]
fn a_link_out_of_the_jail_is_refused() {
    // The case structural checks alone cannot catch: the path has no `..`, no
    // prefix, and looks entirely ordinary. Only physical resolution notices,
    // which is why canonicalisation is not optional.
    //
    // This test is load-bearing in a way the others are not — it is the only
    // one that fails if physical resolution is removed — so it must never
    // skip quietly. A silently skipped test here reads as coverage while
    // testing nothing.
    let (guard, root) = jail_dir();
    let outside = guard.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), "not yours").unwrap();

    link_dir(&outside, &root.join("escape"));

    let jail = Jail::new(&root).unwrap();
    let r = jail.resolve("escape/secret.txt");
    assert!(
        matches!(r, Err(PathRefusal::Escapes { .. })),
        "a linked path must be refused, got {r:?}"
    );
}

/// Create a directory link, by whatever mechanism the platform allows.
///
/// On Windows a real symlink needs Developer Mode or elevation, which CI and
/// most developer machines do not have — but a **directory junction** needs
/// neither, resolves the same way, and is just as good an escape route for an
/// attacker. Falling back to one keeps this test running everywhere instead of
/// vanishing on the platform the harness actually targets.
fn link_dir(target: &Path, link: &Path) {
    #[cfg(windows)]
    {
        if std::os::windows::fs::symlink_dir(target, link).is_ok() {
            return;
        }
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .stdout(std::process::Stdio::null())
            .status()
            .expect("cmd must be available on Windows");
        assert!(
            status.success() && link.exists(),
            "could not create a symlink or a junction; this test must not be skipped"
        );
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).expect("symlink must be creatable");
    }
}

// ------------------------------------------------------------ operations

#[test]
fn reading_a_file_reports_its_contents_as_evidence() {
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::ReadFile {
        path: "src/lib.rs".into(),
    });
    assert!(o.succeeded);
    assert!(o.violations.is_empty());
    assert_eq!(o.evidence["contents"], "fn a() {}\n");
    assert_eq!(o.observed_kind, ActionKind::Read);
}

#[test]
fn writing_creates_parent_directories_and_records_whether_it_overwrote() {
    let (_g, mut e) = exec();
    let first = e.run_op(&Op::WriteFile {
        path: "a/b/new.txt".into(),
        contents: "hello".into(),
    });
    assert!(first.succeeded);
    assert_eq!(first.evidence["overwrote"], false);

    let second = e.run_op(&Op::WriteFile {
        path: "a/b/new.txt".into(),
        contents: "again".into(),
    });
    assert_eq!(second.evidence["overwrote"], true);
    assert_eq!(
        std::fs::read_to_string(e.root().join("a/b/new.txt")).unwrap(),
        "again"
    );
}

#[test]
fn listing_a_directory_is_sorted_and_read_only() {
    let (_g, mut e) = exec();
    e.run_op(&Op::WriteFile {
        path: "src/z.rs".into(),
        contents: "".into(),
    });
    let o = e.run_op(&Op::ListDir { path: "src".into() });
    assert!(o.succeeded);
    assert_eq!(
        o.evidence["entries"],
        serde_json::json!(["lib.rs", "z.rs"])
    );
    assert_eq!(o.observed_kind, ActionKind::Read);
}

#[test]
fn a_missing_file_fails_without_being_a_violation() {
    // Failure and misbehaviour are different things, and conflating them
    // would make ordinary mistakes lexicographically fatal.
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::ReadFile {
        path: "nope.txt".into(),
    });
    assert!(!o.succeeded);
    assert!(o.violations.is_empty());
}

#[test]
fn an_escape_attempt_is_a_violation_not_merely_a_failure() {
    let (_g, mut e) = exec();
    for op in [
        Op::WriteFile {
            path: "../escaped.txt".into(),
            contents: "x".into(),
        },
        Op::ReadFile {
            path: "C:/Windows/win.ini".into(),
        },
        Op::DeleteFile {
            path: "../../thing".into(),
        },
    ] {
        let o = e.run_op(&op);
        assert!(!o.succeeded);
        assert!(is_escape(&o), "{op:?} should record a violation");
    }
}

#[test]
fn a_refused_write_does_not_touch_the_filesystem() {
    let (guard, mut e) = exec();
    let target = guard.path().join("escaped.txt");
    let o = e.run_op(&Op::WriteFile {
        path: "../escaped.txt".into(),
        contents: "should never exist".into(),
    });
    assert!(is_escape(&o));
    assert!(!target.exists(), "the file was written despite the refusal");
}

// ----------------------------------------------------------- subprocesses

#[test]
fn a_command_runs_in_the_sandbox_and_reports_its_exit_code() {
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "git".into(),
        args: vec!["--version".into()],
        timeout_secs: 30,
    });
    assert!(o.succeeded, "evidence: {}", o.evidence);
    assert!(
        o.evidence["stdout"].as_str().unwrap().contains("git version"),
        "stdout should be captured"
    );
}

#[test]
fn a_failing_command_is_reported_as_failure_not_as_an_error() {
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "git".into(),
        args: vec!["rev-parse".into(), "--verify".into(), "nonexistent".into()],
        timeout_secs: 30,
    });
    assert!(!o.succeeded);
    assert_eq!(o.evidence["completion"]["Exited"]["code"].is_null(), false);
}

#[test]
fn a_missing_program_is_unstartable() {
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "definitely-not-a-real-program-xyzzy".into(),
        args: vec![],
        timeout_secs: 5,
    });
    assert!(!o.succeeded);
    assert!(o.evidence["completion"]["Unstartable"].is_object());
}

#[test]
fn a_hanging_command_is_killed_at_its_deadline() {
    let (_g, mut e) = exec();
    let started = std::time::Instant::now();
    // `git` waiting on a pager-free interactive read would be fragile; sleep
    // via the shell is the portable way to hang on purpose here.
    let o = if cfg!(windows) {
        e.run_op(&Op::Run {
            program: "cmd".into(),
            args: vec!["/C".into(), "ping -n 60 127.0.0.1 > NUL".into()],
            timeout_secs: 2,
        })
    } else {
        e.run_op(&Op::Run {
            program: "sh".into(),
            args: vec!["-c".into(), "sleep 60".into()],
            timeout_secs: 2,
        })
    };

    assert!(!o.succeeded);
    assert!(
        o.evidence["completion"]["TimedOut"].is_object(),
        "expected a timeout, got {}",
        o.evidence["completion"]
    );
    assert!(
        started.elapsed().as_secs() < 20,
        "the deadline was not enforced promptly"
    );
}

// ------------------------------------------------------ honest reporting

#[test]
fn exec_reports_the_blast_radius_confinement_actually_provides() {
    // Under PathChecked a subprocess is not confined, so the honest observed
    // blast radius is the machine — regardless of what the agent claimed.
    // Flattering this would make TierMisgrade undetectable for exactly the
    // action class where it matters most.
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "git".into(),
        args: vec!["--version".into()],
        timeout_secs: 30,
    });
    assert_eq!(o.observed_blast_radius, BlastRadius::Machine);
}

#[test]
fn an_understated_action_is_detectable_after_the_fact() {
    let (_g, mut e) = exec();
    let claimed = ProposedAction {
        kind: ActionKind::Read,
        reversibility: Reversibility::Trivial,
        blast_radius: BlastRadius::Episode,
        intent: "just having a look".into(),
        payload: serde_json::json!({
            "do": "run", "program": "git", "args": ["--version"], "timeout_secs": 30
        }),
    };
    let o = e.execute(&claimed);
    assert!(
        o.understated(&claimed),
        "running a subprocess while claiming a sandboxed read must be caught"
    );
}

#[test]
fn a_truthful_action_is_not_flagged() {
    let (_g, mut e) = exec();
    let claimed = ProposedAction {
        kind: ActionKind::Write,
        reversibility: Reversibility::Snapshot,
        blast_radius: BlastRadius::Episode,
        intent: "write a file".into(),
        payload: serde_json::json!({
            "do": "write_file", "path": "notes.txt", "contents": "hello"
        }),
    };
    let o = e.execute(&claimed);
    assert!(o.succeeded);
    assert!(!o.understated(&claimed));
}

#[test]
fn a_malformed_payload_is_refused_rather_than_guessed_at() {
    let (_g, mut e) = exec();
    let o = e.execute(&action(serde_json::json!({"do": "teleport"})));
    assert!(!o.succeeded);
    assert_eq!(o.violations[0].tag, "malformed_action");
}

// ----------------------------------------------------------- confinement

#[test]
fn container_confinement_fails_loudly_rather_than_downgrading() {
    // Silently giving PathChecked to a caller that asked for a container is
    // exactly how an unleashed adversary ends up on the host.
    let (_g, root) = jail_dir();
    let e = Executor::new(&root, Confinement::Container);
    assert!(matches!(e, Err(ExecError::UnsupportedConfinement(_))));
}

#[test]
fn a_jail_must_exist_before_it_can_confine_anything() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("not-created");
    assert!(Executor::new(&missing, Confinement::PathChecked).is_err());
}

// ------------------------------------------------- command normalization

use samaritan_exec::proc::normalize_command;

#[test]
fn a_fused_command_is_split() {
    // What the model actually emitted: {"program": "cargo build", "args": []}.
    let n = normalize_command("cargo build", &[]).expect("should split");
    assert_eq!(n.program, "cargo");
    assert_eq!(n.args, vec!["build"]);
    assert_eq!(n.from, "cargo build");
}

#[test]
fn several_arguments_all_move_across() {
    let n = normalize_command("cargo test --quiet --lib", &[]).unwrap();
    assert_eq!(n.program, "cargo");
    assert_eq!(n.args, vec!["test", "--quiet", "--lib"]);
}

#[test]
fn a_caller_that_supplied_args_is_left_alone() {
    // Supplying args means the schema was understood; the program string is
    // then theirs, spaces and all.
    assert_eq!(normalize_command("my program", &["--x".to_string()]), None);
}

#[test]
fn a_plain_program_is_untouched() {
    assert_eq!(normalize_command("git", &[]), None);
    assert_eq!(normalize_command("  git  ", &[]), None);
}

#[test]
fn a_real_path_containing_spaces_is_not_split() {
    // The case that makes naive splitting wrong. A file that exists is a
    // path, not a fused command, however many spaces it has.
    let dir = tempfile::tempdir().unwrap();
    let spaced = dir.path().join("Program Files");
    std::fs::create_dir_all(&spaced).unwrap();
    let exe = spaced.join("tool.exe");
    std::fs::write(&exe, b"stub").unwrap();

    let p = exe.to_string_lossy().to_string();
    assert!(p.contains(' '), "fixture must contain a space");
    assert_eq!(
        normalize_command(&p, &[]),
        None,
        "an existing path must survive untouched"
    );
}

#[test]
fn a_quoted_executable_keeps_its_spaces() {
    let n = normalize_command(r#""C:\Program Files\Git\bin\git.exe" --version"#, &[]).unwrap();
    assert_eq!(n.program, r"C:\Program Files\Git\bin\git.exe");
    assert_eq!(n.args, vec!["--version"]);
}

#[test]
fn an_empty_program_is_not_invented() {
    assert_eq!(normalize_command("", &[]), None);
    assert_eq!(normalize_command("   ", &[]), None);
}

#[test]
fn the_split_actually_runs_and_is_recorded_in_the_evidence() {
    // End to end: the fused form now works, and the ledger can see that the
    // executor changed the request.
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "git --version".into(),
        args: vec![],
        timeout_secs: 30,
    });
    assert!(o.succeeded, "evidence: {}", o.evidence);
    assert!(o.evidence["stdout"].as_str().unwrap().contains("git version"));
    assert_eq!(o.evidence["normalized"]["from"], "git --version");
    assert_eq!(o.evidence["program"], "git");
    assert_eq!(o.evidence["args"], serde_json::json!(["--version"]));
}

#[test]
fn an_untouched_command_records_no_normalization() {
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "git".into(),
        args: vec!["--version".into()],
        timeout_secs: 30,
    });
    assert!(o.succeeded);
    assert!(
        o.evidence["normalized"].is_null(),
        "nothing was changed, so nothing should be claimed"
    );
}

#[test]
fn splitting_does_not_bypass_the_jail() {
    // A split command is still just a command: same cwd, same routing, same
    // everything. Normalization is a parsing fix, not a privilege change.
    let (_g, mut e) = exec();
    let o = e.run_op(&Op::Run {
        program: "git rev-parse --show-toplevel".into(),
        args: vec![],
        timeout_secs: 30,
    });
    assert_eq!(o.observed_kind, ActionKind::Exec);
    assert_eq!(o.observed_blast_radius, BlastRadius::Machine);
}
