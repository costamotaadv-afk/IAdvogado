python -m pytest& .venv\Scripts\python.exe -m streamlit run app.py --server.port 8999Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*streamlit*"} | Stop-Process -ForceGet-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*streamlit*"} | Stop-Process -Force& .venv\Scripts\python.exe -m streamlit run app.py --server.port 8999& .venv\Scripts\python.exe -m streamlit run app.py --server.port 8999Get-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*streamlit*"} | Stop-Process -ForceGet-Process | Where-Object {$_.ProcessName -eq "python" -and $_.CommandLine -like "*streamlit*"} | Stop-Process -Force& .venv\Scripts\python.exe -m streamlit run app.py --server.port 8999from datetime import datetime, timezone
now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
# ... usar now_str no promptparts = []
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        parts.append(page_text)
text = "\n".join(parts)# Modo interativo
python gerador_parecer_licitacao.py --interativo

# Modo por argumentos
python gerador_parecer_licitacao.py --ente "Maceio" --secretaria "Saude" --processo "123/2026" --interessado "Setor X"