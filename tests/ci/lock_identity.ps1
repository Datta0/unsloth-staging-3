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
    # The arm has to PROVE which rung it took. Both arms agreeing is only evidence if they
    # really were different runs: if the refusal did not take, this measures one resolver
    # against itself and cannot fail.
    $lines += "Write-Output (Get-StudioInstallMutexName -Path '$Path')"
    $lines += 'Write-Output "NATIVE:$($script:StudioFinalPathNativeState)"'
    $file = Join-Path $env:TEMP ("lockid_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -LiteralPath $file -Value ($lines -join "`n") -Encoding utf8
    try { $out = & powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $file 2>&1 }
    finally { Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue }
    $native = (($out | Where-Object { $_ -match "^NATIVE:" }) | Select-Object -First 1)
    return [pscustomobject]@{
        Name   = (($out | Where-Object { $_ -match "^Global\\" }) | Select-Object -First 1)
        Native = ("$native" -replace "^NATIVE:", "")
    }
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
$void = 0
try {
    foreach ($case in $cases.Keys) {
        $native = Get-Name $cases[$case] $false
        $lexical = Get-Name $cases[$case] $true
        $agree = ($native.Name -eq $lexical.Name)
        Write-Host ("{0,-8} native   rung={1} name={2}" -f $case, $native.Native, $native.Name)
        Write-Host ("{0,-8} lexical  rung={1} name={2}  AGREE={3}" -f $case, $lexical.Native, $lexical.Name, $agree)
        if ($native.Native -ne "True" -or $lexical.Native -ne "False") {
            Write-Host "VOID: $case did not run two different rungs (native=$($native.Native) lexical=$($lexical.Native))"
            $void++
        }
        # Repeat the native arm: the same input twice must give the same name, which is
        # the idempotency half of the question.
        $again = Get-Name $cases[$case] $false
        if ($again.Name -ne $native.Name) { Write-Host "UNSTABLE: $case"; $mismatch++ }
        if (-not $agree) { $mismatch++ }
    }
} finally {
    if ($letter) { & subst $letter /D 2>&1 | Out-Null }
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "CASES: $($cases.Keys -join ',')"
Write-Host "DISAGREEMENTS: $mismatch"
Write-Host "VOID_CASES: $void"
if ($void -gt 0) { throw "every case measured one resolver against itself; the refusal did not take" }
# Reported, not thrown. A disagreement on an aliased path is a property the fallback has
# always had; the number is what says whether this change widened it.
