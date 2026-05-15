
Write-Host "Starting Axon Intelligence Stack..." -ForegroundColor Cyan

# Start Telemetry Bridge
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "telemetry_bridge.py" -NoNewWindow

# Start Frontend
Set-Location dashboard
npm run dev
