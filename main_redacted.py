import json
import os

class GeradorParecer:
    def __init__(self, tom='simplificado'):
        self.tom = tom
        self.textos = self.carregar_config()

    def carregar_config(self):
        # Carrega o JSON com os termos de licitação
        # Usando caminho absoluto para evitar erros de diretório atual
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, 'locales', 'licitacoes.json')

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)[self.tom]

    def gerar_parecer(self, empresa, item, resultado_habilitação):
        t = self.textos
        status = t['status_ok'] if resultado_habilitação else t['status_erro']

        # Estrutura do Parecer formatada para o Console do VS Code
        parecer = f"""
        ==================================================
        {t['cabecalho'].upper()}
        ==================================================

        {t['objeto']}
        > Aquisição de: {item}
        > Interessado: {empresa}

        {t['viabilidade']}
        > Parecer: {status}

        {t['conclusao']}
        > Conforme análise, a empresa está apta/inapta.

        --------------------------------------------------
        Referência: {t['lei_ref']}
        ==================================================
        """
        return parecer

# --- TESTE NO TERMINAL DO VS CODE ---
if __name__ == "__main__":
    print("\n--- TESTE 1: TOM SIMPLIFICADO (Para o Diretor da Empresa) ---")
    app_cliente = GeradorParecer(tom='simplificado')
    print(app_cliente.gerar_parecer("Tech Ltda", "Servidores Cloud", True))

    print("\n--- TESTE 2: TOM TÉCNICO (Para o Departamento Jurídico) ---")
    app_juridico = GeradorParecer(tom='tecnico')
    print(app_juridico.gerar_parecer("Tech Ltda", "Servidores Cloud", False))
