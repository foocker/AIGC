param([switch]$Clean)
$ErrorActionPreference = 'Stop'
$editionRoot = $PSScriptRoot
$buildDir = Join-Path $editionRoot 'build\latex'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Push-Location $editionRoot
try {
    if ($Clean) {
        & latexmk -c -outdir=build/latex main_zh.tex
        if ($LASTEXITCODE -ne 0) { throw 'latexmk clean failed' }
    }
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build/latex main_zh.tex
    if ($LASTEXITCODE -ne 0) { throw 'Chinese edition compilation failed' }
    Copy-Item -LiteralPath 'build\latex\main_zh.pdf' -Destination 'principles-diffusion-models-zh.pdf' -Force
} finally {
    Pop-Location
}
