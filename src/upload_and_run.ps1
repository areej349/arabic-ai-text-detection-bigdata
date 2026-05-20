# ============================================================
# upload_and_run.ps1  -  MSBDA-801
# ============================================================
$CENTOS_USER = "hadoop"
$CENTOS_IP   = "192.168.8.82"
$LOCAL_CODE  = "C:\bigdata_project"
$LOCAL_CSV   = "C:\Users\areej\OneDrive\Desktop\AI_Arabic_Detection_ProBDA\Data\raw\raw_combined_abstracts.csv"
$REMOTE_ROOT = "/home/hadoop/bigdata_project"
$REMOTE_RAW  = "/home/hadoop/bigdata_project/data/raw"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  MSBDA-801  Upload and Run" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CentOS : $CENTOS_USER@$CENTOS_IP"
Write-Host "  Local  : $LOCAL_CODE"
Write-Host ""

# ── 1. Create directories ─────────────────────────────────────────────────────
Write-Host "[1/6] Creating directories on CentOS..." -ForegroundColor Yellow
ssh "$CENTOS_USER@$CENTOS_IP" "mkdir -p $REMOTE_RAW /home/hadoop/bigdata_project/data/processed /home/hadoop/bigdata_project/models /home/hadoop/bigdata_project/reports/figures /home/hadoop/bigdata_project/reports/presentations /home/hadoop/bigdata_project/notebooks /home/hadoop/bigdata_project/src"
Write-Host "  Done" -ForegroundColor Green

# ── 2. Upload Python scripts ──────────────────────────────────────────────────
Write-Host "[2/6] Uploading Python scripts..." -ForegroundColor Yellow

$s1 = Join-Path $LOCAL_CODE "phase1_setup.py"
Write-Host "  -> phase1_setup.py"
scp $s1 "${CENTOS_USER}@${CENTOS_IP}:${REMOTE_ROOT}/"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR uploading phase1_setup.py" -ForegroundColor Red
    exit 1
}

$s2 = Join-Path $LOCAL_CODE "phase2_preprocessing.py"
Write-Host "  -> phase2_preprocessing.py"
scp $s2 "${CENTOS_USER}@${CENTOS_IP}:${REMOTE_ROOT}/"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR uploading phase2_preprocessing.py" -ForegroundColor Red
    exit 1
}

$s3 = Join-Path $LOCAL_CODE "phase3_feature_engineering.py"
Write-Host "  -> phase3_feature_engineering.py"
scp $s3 "${CENTOS_USER}@${CENTOS_IP}:${REMOTE_ROOT}/"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR uploading phase3_feature_engineering.py" -ForegroundColor Red
    exit 1
}

Write-Host "  Scripts uploaded OK" -ForegroundColor Green

# ── 3. Upload CSV ─────────────────────────────────────────────────────────────
Write-Host "[3/6] Uploading CSV dataset..." -ForegroundColor Yellow
scp $LOCAL_CSV "${CENTOS_USER}@${CENTOS_IP}:${REMOTE_RAW}/raw_combined_abstracts.csv"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR uploading CSV" -ForegroundColor Red
    exit 1
}
Write-Host "  CSV uploaded OK" -ForegroundColor Green

# ── 4. Run Phase 1 ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[4/6] Running Phase 1..." -ForegroundColor Yellow
ssh "$CENTOS_USER@$CENTOS_IP" "cd /home/hadoop/bigdata_project ; python3 phase1_setup.py 2>&1 | tee logs_phase1.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 1 FAILED - check logs_phase1.txt on CentOS" -ForegroundColor Red
    exit 1
}
Write-Host "  Phase 1 DONE" -ForegroundColor Green

# ── 5. Run Phase 2 ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Running Phase 2..." -ForegroundColor Yellow
ssh "$CENTOS_USER@$CENTOS_IP" "cd /home/hadoop/bigdata_project ; python3 phase2_preprocessing.py 2>&1 | tee logs_phase2.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 2 FAILED - check logs_phase2.txt on CentOS" -ForegroundColor Red
    exit 1
}
Write-Host "  Phase 2 DONE" -ForegroundColor Green

# ── 6. Run Phase 3 ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[6/6] Running Phase 3..." -ForegroundColor Yellow
ssh "$CENTOS_USER@$CENTOS_IP" "cd /home/hadoop/bigdata_project ; python3 phase3_feature_engineering.py 2>&1 | tee logs_phase3.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Phase 3 FAILED - check logs_phase3.txt on CentOS" -ForegroundColor Red
    exit 1
}
Write-Host "  Phase 3 DONE" -ForegroundColor Green

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ALL PHASES COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Logs on CentOS:"
Write-Host "    /home/hadoop/bigdata_project/logs_phase1.txt"
Write-Host "    /home/hadoop/bigdata_project/logs_phase2.txt"
Write-Host "    /home/hadoop/bigdata_project/logs_phase3.txt"
Write-Host "============================================" -ForegroundColor Cyan