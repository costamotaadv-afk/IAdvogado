# 🔒 GUIA DE SEGURANÇA DA API KEY

## ⚠️ IMPORTANTE - LEIA ANTES DE COMPARTILHAR SEU CÓDIGO!

Seu arquivo `.env` contém sua **chave privada da OpenAI** e está configurado para **NÃO ser enviado ao GitHub**.

---

## ✅ Arquivos Seguros para Compartilhar:
- ✅ Todo o código Python (`.py`)
- ✅ Arquivo `.env.example` (template sem chave real)
- ✅ Arquivo `.gitignore` (protege sua chave)
- ✅ README.md, requirements.txt, etc.

## ❌ NUNCA Compartilhe:
- ❌ Arquivo `.env` (contém sua chave real)
- ❌ Screenshots mostrando sua chave completa
- ❌ Logs que possam conter a chave

---

## 🛡️ Como Proteger Sua Chave:

### 1. **Verificar .gitignore**
O arquivo `.gitignore` deve conter:
```
.env
__pycache__/
*.pyc
chroma_db/
```

### 2. **Para Terceiros que Vão Testar:**
Instrua-os a:
1. Clonar o repositório
2. Copiar `.env.example` para `.env`
3. Adicionar a **própria chave** da OpenAI no arquivo `.env`
4. Rodar o aplicativo

### 3. **Verificar se a Chave foi Enviada ao Git:**
```powershell
# Verificar se .env está sendo rastreado
git status

# Se aparecer .env, remova do rastreamento:
git rm --cached .env
git commit -m "Remove .env do repositório"
git push
```

### 4. **Se Sua Chave Foi Exposta:**
1. **Revogue imediatamente** em: https://platform.openai.com/account/api-keys
2. **Crie uma nova chave**
3. **Atualize** o arquivo `.env` local

---

## 📋 Instruções para Terceiros (README.md):

```markdown
## 🔧 Configuração

1. Clone este repositório
2. Instale as dependências:
   ```powershell
   pip install -r requirements.txt
   ```

3. Configure sua chave OpenAI:
   ```powershell
   # Copie o template
   copy .env.example .env
   
   # Edite .env e adicione sua chave:
   OPENAI_API_KEY=sua-chave-aqui
   ```

4. Execute o aplicativo:
   ```powershell
   streamlit run app.py
   ```
```

---

## 🔍 Status de Segurança:

- ✅ Arquivo `.env` criado com sua chave
- ✅ Arquivo `.env.example` criado como template
- ✅ Aplicativo configurado para ler de `.env` ou interface
- ⚠️ **VERIFIQUE** se `.gitignore` contém `.env`

---

## 🚨 Em Caso de Emergência:

Se você acidentalmente compartilhou sua chave:
1. **Acesse:** https://platform.openai.com/account/api-keys
2. **Clique em:** "Revoke" na chave comprometida
3. **Crie** uma nova chave
4. **Atualize** seu `.env` local
5. **Verifique** se não há commits com a chave no histórico do Git

---

**Última atualização:** 27/02/2026
**Status:** ✅ Configurado e protegido
