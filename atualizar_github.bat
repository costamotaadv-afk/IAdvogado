@echo off
chcp 65001 > nul
echo ========================================================
echo       ATUALIZADOR AUTOMATICO GITHUB
echo ========================================================
echo.

echo 1. Adicionando arquivos alterados...
git add .
echo.

echo 2. Criando o commit...
set /p commit_msg="Digite a mensagem do commit (ex: Ajustes no layout): "
if "%commit_msg%"=="" set commit_msg="Atualizacao automatica"
git commit -m "%commit_msg%"
echo.

echo 3. Enviando para o GitHub...
git push
echo.

if %errorlevel% neq 0 (
    echo [ERRO] Nao foi possivel enviar para o GitHub.
    echo Verifique se voce ja configurou o repositorio remoto.
    echo (git remote pair add origin URL_DO_SEU_REPO)
    pause
    exit /b
)

echo [SUCESSO] Seu codigo esta atualizado no GitHub!
timeout /t 5
