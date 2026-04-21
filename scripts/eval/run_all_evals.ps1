# run_all_evals.ps1
# Runs the full two-phase ablation eval pipeline sequentially.
# No user interaction required; exits on first failure.
#
# Usage (from repo root):
#   .\scripts\eval\run_all_evals.ps1                           # full run
#   .\scripts\eval\run_all_evals.ps1 -SkipRetriever            # generator only (all retriever results exist)
#   .\scripts\eval\run_all_evals.ps1 -Limit 50                 # smoke-test with 50 questions
#   .\scripts\eval\run_all_evals.ps1 -SkipGenerator            # retriever axes only
#   powershell -ExecutionPolicy Bypass -File scripts\eval\run_all_evals.ps1 -SkipRetriever
#
# Requirements:
#   - Conda/venv with project deps activated before running.
#   - OPENROUTER_API_KEY in .env (needed for generator phase + BERTScore model).
#   - Ollama running locally for gemma4 generator models.

param(
    [int]$Limit     = 0,        # 0 = full dataset
    [switch]$SkipGenerator,     # skip Phase 2
    [switch]$SkipRetriever,     # skip ALL Phase 1 axes (use when retriever results already exist)
    [switch]$SkipEmbedding,     # skip embedding axis (slow, needs all ingests)
    [switch]$SkipChunking       # skip chunking axis (needs langchain ingest)
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..") # repo root

$python = "python"
$runner = "scripts\eval\run_eval.py"

# Build shared limit flag
$limitFlag = if ($Limit -gt 0) { @("--limit", "$Limit") } else { @() }

function Run-Step {
    param([string]$Label, [string[]]$CmdArgs)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    & $python $runner @CmdArgs @limitFlag
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Label (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "DONE: $Label" -ForegroundColor Green
}

# ------------------------------------------------------------------
# Phase 1a: Search-mode axis  (semantic / keyword / hybrid)
#   Results already exist: retriever_search_20260408*.md
# ------------------------------------------------------------------
if (-not $SkipRetriever) {
    $label1a = "Phase 1 - Retriever: search axis"
    Run-Step $label1a @("--phase", "retriever", "--axis", "search")
} else {
    Write-Host "Skipping all retriever phases (-SkipRetriever)" -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# Phase 1b: RRF alpha sweep  (hybrid only, varies alpha 0.5->0.9)
#   Results already exist: retriever_rrf_20260410*.md
# ------------------------------------------------------------------
if (-not $SkipRetriever) {
    $label1b = "Phase 1 - Retriever: RRF alpha sweep"
    Run-Step $label1b @("--phase", "retriever", "--axis", "rrf")
}

# ------------------------------------------------------------------
# Phase 1c: Embedding axis  (results exist: retriever_embedding_20260415*.md)
# ------------------------------------------------------------------
if (-not $SkipRetriever -and -not $SkipEmbedding) {
    $label1c = "Phase 1 - Retriever: embedding axis"
    Run-Step $label1c @("--phase", "retriever", "--axis", "embedding")
} elseif (-not $SkipRetriever) {
    Write-Host "Skipping embedding axis (-SkipEmbedding)" -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# Phase 1d: Chunking axis  (needs langchain ingest)
# ------------------------------------------------------------------
if (-not $SkipRetriever -and -not $SkipChunking) {
    $label1d = "Phase 1 - Retriever: chunking axis"
    Run-Step $label1d @("--phase", "retriever", "--axis", "chunking")
} elseif (-not $SkipRetriever) {
    Write-Host "Skipping chunking axis (-SkipChunking)" -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# Phase 2: Generator  (locked to hybrid alpha=0.80, nomic+langchain - best R@10)
# ------------------------------------------------------------------
if (-not $SkipGenerator) {
    $label2   = "Phase 2 - Generator: all LLMs"
    $genArgs  = @("--phase", "generator", "--embedding", "nomic-ai/nomic-embed-text-v1.5", "--search-mode", "hybrid", "--chunking", "langchain")
    Run-Step $label2 $genArgs
} else {
    Write-Host "Skipping generator phase (-SkipGenerator)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All evals complete." -ForegroundColor Green
Write-Host "  Results in: data/eval/results/" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
