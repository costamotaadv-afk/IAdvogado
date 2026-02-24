import subprocess
import sys
import datetime

def run_command(command, description):
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ ERRO: {description} falhou!")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print(f"✅ SUCESSO: {description}")
    return True

def main():
    print("="*50)
    print("🚀 INICIANDO ROTINA DE TESTE E DEPLOY")
    print("="*50)

    # 1. Rodar os testes
    # Usando o executável do ambiente virtual
    python_exe = r".venv\Scripts\python.exe"
    test_command = f"{python_exe} -m pytest"
    
    if not run_command(test_command, "Executando testes com pytest"):
        print("\n⚠️ Deploy cancelado porque os testes falharam. Corrija os erros e tente novamente.")
        sys.exit(1)

    # 2. Adicionar arquivos ao Git
    if not run_command("git add .", "Adicionando arquivos ao Git"):
        sys.exit(1)

    # 3. Fazer o Commit
    # Verifica se há algo para commitar
    status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if not status.stdout.strip():
        print("\nℹ️ Nenhuma alteração nova para salvar no Git.")
        sys.exit(0)

    commit_msg = f"Auto-deploy após testes passarem: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if not run_command(f'git commit -m "{commit_msg}"', "Criando commit"):
        sys.exit(1)

    # 4. Fazer o Push para o GitHub
    if not run_command("git push origin main", "Enviando para o GitHub (Push)"):
        print("\n⚠️ Dica: Se for a primeira vez, certifique-se de ter configurado o repositório remoto (git remote add origin ...)")
        sys.exit(1)

    print("\n" + "="*50)
    print("🎉 DEPLOY CONCLUÍDO COM SUCESSO! Todos os testes passaram e o código está no GitHub.")
    print("="*50)

if __name__ == "__main__":
    main()
