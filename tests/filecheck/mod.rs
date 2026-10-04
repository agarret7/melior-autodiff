use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Runs LLVM `FileCheck` over `ir` using the `CHECK` lines in `check_file`, a path relative to
/// `tests/`. `prefix` selects `<PREFIX>:` lines instead of `CHECK:`.
pub fn filecheck(ir: &str, check_file: &str, prefix: Option<&str>) {
    let check_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(check_file);
    let mut command = Command::new(filecheck_binary());
    command
        .arg(&check_path)
        .arg("--input-file=-")
        .arg("--dump-input=fail");
    if let Some(prefix) = prefix {
        command.arg(format!("--check-prefix={prefix}"));
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to run FileCheck");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(ir.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "FileCheck failed for {check_file}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn filecheck_binary() -> PathBuf {
    let prefix = std::env::var("MLIR_SYS_230_PREFIX").unwrap_or_else(|_| "/usr/lib/llvm-23".into());
    Path::new(&prefix).join("bin/FileCheck")
}
