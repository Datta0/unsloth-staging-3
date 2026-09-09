# Does a run that resolves paths natively agree with one that falls back to the lexical
# resolver, on the identity the install lock is named after?
#
# This is the question PR 10540 widens: refusing the emit under an audit-mode App Control
# policy sends more machines down the lexical path, and two concurrent installs that
# disagree about a path's identity take different locks and do not exclude each other.
# It cannot be measured off Windows, where both arms degrade and every answer matches for
# the wrong reason. Three inputs, because they are the three ways the two resolvers are
# documented to differ: a plain path, an 8.3 alias, and a SUBST drive.
$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot | Split-Path | Split-Path
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $repoRoot "install.ps1"), [ref]$null, [ref]$null)
function Src($name) {
    $f = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $args[0].Name -eq $name }, $true)
    if ($f.Count -eq 0) { throw "missing $name" }
    return $f[0].Extent.Text
}
$chain = @("Write-StudioLine","Write-StudioFinalPathDegraded","Test-StudioCanDefineNativeTypes",
           "New-StudioDynamicAssembly","New-StudioEmittedNativeType",
           "Initialize-StudioFinalPathNativeType","Get-StudioNativeFinalPath",
           "Resolve-StudioLinkTarget","Get-StudioSubstTarget","Get-StudioLexicalPath",
           "Resolve-StudioFinalPathInfo","Get-StudioFinalPath","Get-StudioPathHash",
           "Get-StudioInstallMutexName")

function Get-Name([string]$Path, [bool]$Refuse) {
    # A fresh child each time: these functions cache their answer in script scope on
    # purpose, so the two arms cannot share a session without one poisoning the other.
    $lines = @('$ErrorActionPreference = "Stop"', '$script:StudioStdoutRedirected = $true')
    $lines += ($chain | ForEach-Object { Src $_ })
    if ($Refuse) { $lines += 'function Test-StudioCanDefineNativeTypes { return $false }' }
    $lines += "Write-Output (Get-StudioInstallMutexName -Path '$Path')"
    $file = Join-Path $env:TEMP ("lockid_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $file -Value ($lines -join "`n") -Encoding utf8
    try { $out = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $file 2>&1 }
    finally { Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue }
    return (($out | Where-Object { $_ -match "^Global\\" }) | Select-Object -First 1)
}

$root = Join-Path $env:TEMP ("lockid " + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $root | Out-Null
$cases = @{ plain = $root }

# 8.3 alias, if the volume still generates them. Not all do, so an absent alias skips
# rather than passes.
$short = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($root).ShortPath
if ($short -and $short -ne $root) { $cases["short83"] = $short }

# SUBST drive pointing at the same directory.
$letter = $null
foreach ($c in [char[]]"XYWVU") {
    if (-not (Test-Path "${c}:")) { $letter = "${c}:"; break }
}
if ($letter) {
    & subst $letter $root 2>&1 | Out-Null
    if (Test-Path $letter) { $cases["subst"] = "$letter\" }
}

$mismatch = 0
try {
    foreach ($case in $cases.Keys) {
        $native = Get-Name $cases[$case] $false
        $lexical = Get-Name $cases[$case] $true
        $agree = ($native -eq $lexical)
        Write-Host ("{0,-8} native={1}" -f $case, $native)
        Write-Host ("{0,-8} lexical={1}  AGREE={2}" -f $case, $lexical, $agree)
        # Repeat the native arm: the same input twice must give the same name, which is
        # the idempotency half of the question.
        $again = Get-Name $cases[$case] $false
        if ($again -ne $native) { Write-Host "UNSTABLE: $case"; $mismatch++ }
        if (-not $agree) { $mismatch++ }
    }
} finally {
    if ($letter) { & subst $letter /D 2>&1 | Out-Null }
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "CASES: $($cases.Keys -join ',')"
Write-Host "DISAGREEMENTS: $mismatch"
# Reported, not thrown. A disagreement on an aliased path is a property the fallback has
# always had; the number is what says whether this change widened it.
