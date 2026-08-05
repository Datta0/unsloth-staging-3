//! End-to-end before/after for PR #7943.
//!
//! Reproduces what install.rs actually receives on Windows: tauri-utils calls
//! `canonicalize()` on the running exe (starting_binary.rs), and `resource_dir()`
//! returns that path's parent, so the script path inherits the `\\?\` prefix.
//! Prints the pre-PR spelling and the post-PR spelling so a PowerShell step can
//! hand both to the real Windows PowerShell 5.1.

use std::path::{Path, PathBuf};

#[cfg(windows)]
fn powershell_script_path(path: &Path) -> PathBuf {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    // Everything after `\\?\` reaches the object manager, which is case insensitive.
    fn is(unit: Option<&u16>, ascii: u8) -> bool {
        unit.is_some_and(|value| *value < 128 && (*value as u8).eq_ignore_ascii_case(&ascii))
    }

    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let verbatim: Vec<u16> = r"\\?\".encode_utf16().collect();
    if !wide.starts_with(&verbatim) {
        return path.to_path_buf();
    }
    let rest = &wide[verbatim.len()..];

    let is_drive = rest.first().is_some_and(|value| {
        (b'A' as u16..=b'Z' as u16).contains(value) || (b'a' as u16..=b'z' as u16).contains(value)
    }) && rest.get(1) == Some(&(b':' as u16))
        && rest.get(2) == Some(&(b'\\' as u16));

    let normalized = if is(rest.first(), b'U')
        && is(rest.get(1), b'N')
        && is(rest.get(2), b'C')
        && rest.get(3) == Some(&(b'\\' as u16))
    {
        let mut value: Vec<u16> = r"\\".encode_utf16().collect();
        value.extend_from_slice(&rest[4..]);
        value
    } else if is_drive {
        rest.to_vec()
    } else {
        return path.to_path_buf();
    };

    // Only the verbatim form addresses a path past MAX_PATH; stripping it there
    // would trade an authorization error for a "path too long" one. MAX_PATH
    // counts the terminating NUL, so 259 units is the longest legacy path.
    if normalized.len() >= 260 {
        return path.to_path_buf();
    }

    PathBuf::from(OsString::from_wide(&normalized))
}

#[cfg(not(windows))]
fn powershell_script_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

fn main() {
    let dir = std::env::var("E2E_DIR").expect("E2E_DIR not set");
    let script = PathBuf::from(&dir).join("install.ps1");
    let canon = std::fs::canonicalize(&script).expect("canonicalize failed");
    println!("TAURI_LIKE_PATH={}", canon.display());
    println!("AFTER_PR_PATH={}", powershell_script_path(&canon).display());
}
