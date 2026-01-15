$ErrorActionPreference = 'Stop'

Write-Host 'Starting realtime collector...'
Start-Process -WindowStyle Minimized -FilePath '.\venv\Scripts\python.exe' -ArgumentList 'monitoring_system/realtime_collector.py'

Write-Host 'Starting Streamlit...'
Start-Process -WindowStyle Minimized -FilePath '.\venv\Scripts\streamlit.exe' -ArgumentList 'run', 'monitoring_system/app.py', '--server.port', '6114'

Write-Host 'Started.'
