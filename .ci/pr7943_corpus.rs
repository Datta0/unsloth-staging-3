//! Exhaustive edge-case corpus for PR #7943's `powershell_script_path`.
//!
//! On Windows this runs the PR's function verbatim (real `OsStr`/`Path`, real
//! filesystem probes). On Linux/macOS it runs an OS-independent port of the same
//! index arithmetic so the two can be compared. Exits non-zero on a real failure.

use std::path::{Path, PathBuf};

// ===========================================================================
// Copied verbatim from studio/src-tauri/src/install.rs at this commit.
// ===========================================================================
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

    // Everything after `\\?\` reaches the object manager, which is case insensitive.
    fn is(unit: Option<&u16>, ascii: u8) -> bool {
        unit.is_some_and(|value| *value < 128 && (*value as u8).eq_ignore_ascii_case(&ascii))
    }

    let wide: Vec<u16> = path.to_string_lossy().encode_utf16().collect();
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

    PathBuf::from(String::from_utf16_lossy(&normalized))
}

fn norm(p: &str) -> String {
    powershell_script_path(Path::new(p)).to_string_lossy().into_owned()
}

fn main() {
    let mut fails = 0usize;
    let mut n = 0usize;

    let mut check = |name: &str, input: &str, expect: &str| {
        n += 1;
        let got = norm(input);
        if got != expect {
            fails += 1;
            println!("  FAIL {name}\n       in={input:?}\n      got={got:?}\n     want={expect:?}");
        }
    };

    println!("== Table cases ==");
    // the PR's own four
    check("pr/verbatim-drive", r"\\?\C:\Users\Owner\install.ps1", r"C:\Users\Owner\install.ps1");
    check("pr/verbatim-unc", r"\\?\UNC\server\share\install.ps1", r"\\server\share\install.ps1");
    check("pr/ordinary", r"C:\Users\Owner\install.ps1", r"C:\Users\Owner\install.ps1");
    check("pr/volume-guid", r"\\?\Volume{1234}\install.ps1", r"\\?\Volume{1234}\install.ps1");
    // the field-report path shape: spaces + parentheses
    check(
        "field/real-install-dir",
        r"\\?\C:\Users\Owner\AppData\Local\Unsloth Studio (Desktop)\install.ps1",
        r"C:\Users\Owner\AppData\Local\Unsloth Studio (Desktop)\install.ps1",
    );
    // drive letters
    check("drive/lower", r"\\?\c:\x\install.ps1", r"c:\x\install.ps1");
    check("drive/A", r"\\?\A:\i.ps1", r"A:\i.ps1");
    check("drive/Z", r"\\?\Z:\i.ps1", r"Z:\i.ps1");
    check("drive/digit", r"\\?\1:\i.ps1", r"\\?\1:\i.ps1");
    // degenerate
    check("degen/empty", "", "");
    check("degen/bare-prefix", r"\\?\", r"\\?\");
    check("degen/no-colon", r"\\?\C", r"\\?\C");
    check("degen/colon-only", r"\\?\C:", r"\\?\C:");
    check("degen/drive-root", r"\\?\C:\", r"C:\");
    check("degen/unc-bare", r"\\?\UNC\", r"\\");
    check("degen/unc-server-only", r"\\?\UNC\server", r"\\server");
    check("degen/relative", r"install.ps1", r"install.ps1");
    // pass-through
    check("thru/device-ns", r"\\.\C:\i.ps1", r"\\.\C:\i.ps1");
    check("thru/globalroot", r"\\?\GLOBALROOT\Device\HarddiskVolume1\i.ps1", r"\\?\GLOBALROOT\Device\HarddiskVolume1\i.ps1");
    check("thru/plain-unc", r"\\server\share\i.ps1", r"\\server\share\i.ps1");
    check("thru/fwd-slash-verbatim", r"\\?\C:/Users/i.ps1", r"\\?\C:/Users/i.ps1");
    // unicode
    check("uni/cyrillic", r"\\?\C:\Users\Даниэль\i.ps1", r"C:\Users\Даниэль\i.ps1");
    check("uni/cjk", r"\\?\C:\Users\日本語\i.ps1", r"C:\Users\日本語\i.ps1");
    check("uni/emoji", r"\\?\C:\Users\🦥\i.ps1", r"C:\Users\🦥\i.ps1");
    println!("  {n} cases, {fails} failed");

    println!("\n== UNC token case permutations ==");
    let mut missed = 0;
    for bits in 0u8..8 {
        let tok: String = "UNC".chars().enumerate()
            .map(|(i, c)| if bits >> i & 1 == 1 { c.to_ascii_lowercase() } else { c })
            .collect();
        let inp = format!(r"\\?\{tok}\server\share\i.ps1");
        let got = norm(&inp);
        let ok = got == r"\\server\share\i.ps1";
        if !ok { missed += 1; }
        println!("  {tok:<5} -> {:<42} {}", got, if ok { "normalized" } else { "LEFT VERBATIM" });
    }
    println!("  {missed}/8 spellings left unnormalized (expected 0)");
    fails += missed;

    println!("\n== Invariants over a generated corpus ==");
    let prefixes = [r"\\?\", r"\\.\", r"\\", "", r"\", r"\\?", r"\?\"];
    let bodies = ["C:\\", "c:\\", "C:", "C", "UNC\\", "unc\\", "Unc\\", "Volume{a}\\",
                  "GLOBALROOT\\", "server\\share\\", "", "1:\\", ":\\"];
    let tails = ["install.ps1", "", "a\\install.ps1", "a b (x)\\install.ps1"];
    let mut corpus = Vec::new();
    for p in prefixes { for b in bodies { for t in tails { corpus.push(format!("{p}{b}{t}")); } } }
    let (mut idem, mut touched_nonverbatim) = (0, 0);
    for c in &corpus {
        let once = norm(c);
        if norm(&once) != once { idem += 1; println!("  NOT IDEMPOTENT: {c:?} -> {once:?}"); }
        if !c.starts_with(r"\\?\") && once != *c {
            touched_nonverbatim += 1;
            println!("  NON-VERBATIM MODIFIED: {c:?} -> {once:?}");
        }
    }
    println!("  corpus={} idempotence_failures={idem} non_verbatim_modified={touched_nonverbatim}",
             corpus.len());
    fails += idem + touched_nonverbatim;

    println!("\n== MAX_PATH behaviour ==");
    for total in [200usize, 259, 260, 261, 300, 400] {
        let base = r"\\?\C:\";
        let fill = total.saturating_sub(base.len() + 12);
        let inp = format!(r"{base}{}\install.ps1", "a".repeat(fill));
        let out = norm(&inp);
        let over = out.chars().count() > 260;
        println!("  in={:<4} out={:<4} de-verbatimized={:<5} out>MAX_PATH={}",
                 inp.chars().count(), out.chars().count(), out != inp, over);
        // A stripped path must never end up longer than MAX_PATH.
        if over && out != inp {
            println!("  FAIL: de-verbatimized past MAX_PATH");
            fails += 1;
        }
    }

    // Real filesystem probes: only meaningful on Windows.
    #[cfg(windows)]
    {
        use std::fs;
        println!("\n== Real Windows filesystem probes ==");
        let tmp = std::env::var("RUNNER_TEMP").unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        let dir = PathBuf::from(&tmp).join("Unsloth Studio (Desktop)");
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("install.ps1");
        fs::write(&script, b"Write-Output 'ok'\n").unwrap();

        let canon = fs::canonicalize(&script).unwrap();
        println!("  canonicalize() -> {canon:?}");
        println!("  canonicalize is verbatim = {}",
                 canon.to_string_lossy().starts_with(r"\\?\"));
        let fixed = powershell_script_path(&canon);
        println!("  after powershell_script_path -> {fixed:?}");
        println!("  fixed path still opens = {}", fs::metadata(&fixed).is_ok());
        println!("  fixed path is no longer verbatim = {}",
                 !fixed.to_string_lossy().starts_with(r"\\?\"));
        if fs::metadata(&fixed).is_err() {
            println!("  FAIL: normalized path does not resolve on the real filesystem");
            fails += 1;
        }

        // >MAX_PATH probe: can the de-verbatimized long path still be opened?
        let mut deep = PathBuf::from(&tmp);
        for _ in 0..12 { deep.push("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); }
        let verbatim_deep = PathBuf::from(format!(r"\\?\{}", deep.display()));
        match fs::create_dir_all(&verbatim_deep) {
            Ok(_) => {
                let long_script = verbatim_deep.join("install.ps1");
                let _ = fs::write(&long_script, b"x");
                let stripped = powershell_script_path(&long_script);
                println!("  long path len={} -> stripped len={}",
                         long_script.to_string_lossy().chars().count(),
                         stripped.to_string_lossy().chars().count());
                println!("  long verbatim opens = {}", fs::metadata(&long_script).is_ok());
                println!("  long stripped opens = {} (false here = the MAX_PATH concern is real)",
                         fs::metadata(&stripped).is_ok());
            }
            Err(e) => println!("  could not build a >MAX_PATH dir: {e}"),
        }
    }

    println!("\n== RESULT: {} ==", if fails == 0 { "PASS" } else { "FAIL" });
    if fails > 0 {
        std::process::exit(1);
    }
}
