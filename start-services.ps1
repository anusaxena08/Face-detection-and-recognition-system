$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source

Write-Host "Using Python: $python"

$backend = Start-Process -FilePath $python `
    -ArgumentList '-m', 'uvicorn', 'backend.api:app', '--host', '127.0.0.1', '--port', '8000' `
    -WorkingDirectory $projectRoot `
    -PassThru

$frontend = Start-Process -FilePath $python `
    -ArgumentList '-m', 'streamlit', 'run', 'frontend/app.py', '--server.address=127.0.0.1', '--server.port=8501' `
    -WorkingDirectory $projectRoot `
    -PassThru

Write-Host "Backend PID : $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Frontend URL: http://localhost:8501"
Write-Host "Backend URL : http://localhost:8000"