param(
    [ValidateSet('all', 'seedance', 'theory', 'full')]
    [string]$Target = 'all'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    $jobs = @()
    if ($Target -in @('all', 'theory')) { $jobs += 'AIGCTheory' }
    if ($Target -in @('all', 'seedance')) { $jobs += 'Seedance25' }
    if ($Target -in @('all', 'full')) { $jobs += 'StabelDiffusion' }
    foreach ($bookName in $jobs) {
        $buildDirectory = Join-Path 'build' $bookName
        New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
        & latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error "-outdir=$buildDirectory" "$bookName.tex"
        if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $bookName" }
        Copy-Item -LiteralPath (Join-Path $buildDirectory "$bookName.pdf") -Destination "$bookName.pdf" -Force
    }
} finally {
    Pop-Location
}
exit 0
