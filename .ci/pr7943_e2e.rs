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

    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let verbatim_unc: Vec<u16> = r"\\?\UNC\".encode_utf16().collect();
    let verbatim: Vec<u16> = r"\\?\".encode_utf16().collect();

    let normalized = if wide.starts_with(&verbatim_unc) {
        let mut value: Vec<u16> = r"\\".encode_utf16().collect();
        value.extend_from_slice(&wide[verbatim_unc.len()..]);
        value
    } else {
        let drive = wide.get(verbatim.len()).copied();
        let is_ascii_drive = drive.is_some_and(|value| {
            (b'A' as u16..=b'Z' as u16).contains(&value)
                || (b'a' as u16..=b'z' as u16).contains(&value)
        });
        if wide.starts_with(&verbatim)
            && is_ascii_drive
            && wide.get(verbatim.len() + 1) == Some(&(b':' as u16))
            && wide.get(verbatim.len() + 2) == Some(&(b'\\' as u16))
        {
            wide[verbatim.len()..].to_vec()
        } else {
            return path.to_path_buf();
        }
    };

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
