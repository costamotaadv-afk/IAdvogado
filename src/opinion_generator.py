import os
from datetime import datetime, timezone
import yaml
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict, Counter
import json
from src.legal_classifier import classify_legal_case
from src.template_manager import load_template

# ========== SISTEMA DE APRENDIZADO LINGUÍSTICO ==========

# Memória persistente de padrões aprendidos
_linguistic_memory = {
    'word_relations': defaultdict(Counter),  # palavra -> {palavra_relacionada: frequência}
    'sentence_structures': defaultdict(int),  # padrão -> frequência
    'reasoning_patterns': defaultdict(list),  # tipo_raciocínio -> [exemplos]
    'argument_styles': defaultdict(int),  # tipo_argumento -> frequência
    'writing_styles': defaultdict(list),  # característica -> [exemplos]
}

def analyze_word_relations(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    APRENDIZADO: Relações entre palavras (co-ocorrência e semântica).
    
    Analisa quais palavras aparecem juntas com frequência, permitindo que o agente
    "aprenda" vocabulário contextual e relações semânticas.
    
    Exemplo: "edital" frequentemente aparece com "licitação", "pregão", "classificação"
    
    Args:
        text (str): Texto para análise
        
    Returns:
        Dict com relações identificadas e aprendidas
    """
    global _linguistic_memory
    
    # Tokenização básica
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords comuns
    stopwords = {'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na',
                 'e', 'ou', 'para', 'por', 'com', 'um', 'uma', 'que', 'se', 'é', 'são'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    relations = []
    
    # Análise de co-ocorrência (janela de 5 palavras)
    window_size = 5
    for i, word in enumerate(words):
        # Palavras próximas (contexto)
        context_start = max(0, i - window_size)
        context_end = min(len(words), i + window_size + 1)
        context_words = words[context_start:i] + words[i+1:context_end]
        
        # Registra relações na memória
        for context_word in context_words:
            _linguistic_memory['word_relations'][word][context_word] += 1
            relations.append((word, context_word))
    
    # Identifica as relações mais fortes aprendidas
    learned_relations = []
    for word, related in _linguistic_memory['word_relations'].items():
        if len(related) > 0:
            # Top 3 palavras mais relacionadas
            top_related = related.most_common(3)
            learned_relations.append((word, [r[0] for r in top_related]))
    
    return {
        'current_relations': relations[:20],  # Relações no texto atual
        'learned_vocabulary': len(_linguistic_memory['word_relations']),
        'top_learned_relations': learned_relations[:10],
        'learning_note': 'O agente aprende quais palavras aparecem juntas e fortalece conexões semânticas'
    }

def analyze_sentence_structure(text: str) -> Dict[str, any]:
    """
    APRENDIZADO: Estrutura de frases (padrões sintáticos).
    
    Identifica padrões como:
    - Sujeito-Verbo-Objeto
    - Frases condicionais (se...então)
    - Estruturas causais (porque, portanto, assim)
    - Frases interrogativas
    
    O agente aprende estruturas comuns e pode reproduzi-las.
    
    Args:
        text (str): Texto para análise
        
    Returns:
        Dict com estruturas identificadas e aprendidas
    """
    global _linguistic_memory
    
    sentences = re.split(r'[.!?]+', text)
    structures_found = []
    
    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
            
        sentence_lower = sentence.lower().strip()
        
        # Detecta padrões estruturais
        pattern = 'desconhecido'
        
        # Estrutura condicional
        if 'se ' in sentence_lower and any(marker in sentence_lower for marker in ['então', 'logo', ',']):
            pattern = 'condicional:se-então'
            
        # Estrutura causal
        elif any(marker in sentence_lower for marker in ['porque', 'pois', 'uma vez que', 'visto que']):
            pattern = 'causal:causa-efeito'
            
        # Estrutura conclusiva
        elif any(marker in sentence_lower for marker in ['portanto', 'assim', 'logo', 'consequentemente', 'dessa forma']):
            pattern = 'conclusiva:premissa-conclusão'
            
        # Estrutura interrogativa
        elif sentence_lower.startswith(('qual', 'quais', 'como', 'por que', 'quando', 'onde', 'quanto')):
            pattern = 'interrogativa:pergunta-direta'
            
        # Estrutura imperativa
        elif any(verb in sentence_lower.split()[:2] for verb in ['deve', 'deverá', 'é necessário', 'requer', 'exige']):
            pattern = 'imperativa:comando-obrigação'
            
        # Estrutura descritiva (padrão)
        else:
            pattern = 'descritiva:sujeito-predicado'
        
        # Aprende o padrão
        _linguistic_memory['sentence_structures'][pattern] += 1
        structures_found.append({
            'pattern': pattern,
            'example': sentence.strip()[:100] + '...' if len(sentence) > 100 else sentence.strip()
        })
    
    # Estatísticas de aprendizado
    total_patterns = sum(_linguistic_memory['sentence_structures'].values())
    pattern_distribution = {
        pattern: (count / total_patterns * 100) if total_patterns > 0 else 0
        for pattern, count in _linguistic_memory['sentence_structures'].items()
    }
    
    return {
        'structures_in_text': structures_found[:10],
        'total_patterns_learned': total_patterns,
        'pattern_distribution': pattern_distribution,
        'most_common_structure': max(_linguistic_memory['sentence_structures'], 
                                     key=_linguistic_memory['sentence_structures'].get) if _linguistic_memory['sentence_structures'] else 'Nenhum',
        'learning_note': 'O agente identifica e aprende estruturas sintáticas para reproduzir padrões naturais'
    }

def analyze_reasoning_patterns(text: str) -> Dict[str, any]:
    """
    APRENDIZADO: Padrões de raciocínio (lógica argumentativa).
    
    Identifica como ideias são conectadas:
    - Raciocínio dedutivo (regra geral → caso específico)
    - Raciocínio indutivo (casos específicos → generalização)
    - Raciocínio analógico (comparação por semelhança)
    - Raciocínio causal (causa → efeito)
    
    Args:
        text (str): Texto para análise
        
    Returns:
        Dict com padrões de raciocínio identificados
    """
    global _linguistic_memory
    
    patterns_detected = []
    
    text_lower = text.lower()
    
    # Marcadores de raciocínio dedutivo
    deductive_markers = ['segundo a lei', 'conforme', 'de acordo com', 'nos termos', 
                        'aplicando-se', 'por força de', 'em observância']
    if any(marker in text_lower for marker in deductive_markers):
        patterns_detected.append('dedutivo:regra→caso')
        _linguistic_memory['reasoning_patterns']['dedutivo'].append(text[:200])
    
    # Marcadores de raciocínio indutivo
    inductive_markers = ['observa-se que', 'verifica-se que', 'constata-se', 
                        'diversos casos', 'exemplos demonstram', 'evidências indicam']
    if any(marker in text_lower for marker in inductive_markers):
        patterns_detected.append('indutivo:casos→regra')
        _linguistic_memory['reasoning_patterns']['indutivo'].append(text[:200])
    
    # Marcadores de raciocínio analógico
    analogical_markers = ['semelhante a', 'assim como', 'da mesma forma', 
                         'analogamente', 'similar', 'comparável']
    if any(marker in text_lower for marker in analogical_markers):
        patterns_detected.append('analógico:comparação')
        _linguistic_memory['reasoning_patterns']['analógico'].append(text[:200])
    
    # Marcadores de raciocínio causal
    causal_markers = ['porque', 'pois', 'uma vez que', 'visto que', 'em razão de',
                     'por conseguinte', 'resulta em', 'leva a', 'implica']
    if any(marker in text_lower for marker in causal_markers):
        patterns_detected.append('causal:causa→efeito')
        _linguistic_memory['reasoning_patterns']['causal'].append(text[:200])
    
    # Estatísticas de padrões aprendidos
    reasoning_stats = {
        tipo: len(exemplos) for tipo, exemplos in _linguistic_memory['reasoning_patterns'].items()
    }
    
    return {
        'patterns_detected': patterns_detected,
        'reasoning_statistics': reasoning_stats,
        'total_reasoning_examples': sum(reasoning_stats.values()),
        'learning_note': 'O agente aprende como construir argumentos lógicos e estruturar raciocínios'
    }

def analyze_argument_styles(text: str) -> Dict[str, any]:
    """
    APRENDIZADO: Formas de argumentar (tipos de argumento).
    
    Identifica estratégias argumentativas:
    - Argumento por autoridade (citação de especialistas, leis, jurisprudência)
    - Argumento por evidência (dados, fatos, exemplos concretos)
    - Argumento por definição (conceituação, esclarecimento de termos)
    - Argumento por consequência (implicações, resultados)
    - Argumento por comparação (analogias, diferenciações)
    
    Args:
        text (str): Texto para análise
        
    Returns:
        Dict com estilos argumentativos identificados
    """
    global _linguistic_memory
    
    text_lower = text.lower()
    arguments_found = []
    
    # Argumento por autoridade
    authority_markers = ['segundo', 'conforme', 'de acordo com', 'jurisprudência', 
                        'doutrina', 'tribunal', 'stf', 'stj', 'tcu', 'lei', 'artigo']
    if any(marker in text_lower for marker in authority_markers):
        arguments_found.append('por-autoridade')
        _linguistic_memory['argument_styles']['por-autoridade'] += 1
    
    # Argumento por evidência
    evidence_markers = ['dados', 'pesquisa', 'estudo', 'análise', 'estatística', 
                       'comprova', 'demonstra', 'evidencia', 'constata-se']
    if any(marker in text_lower for marker in evidence_markers):
        arguments_found.append('por-evidência')
        _linguistic_memory['argument_styles']['por-evidência'] += 1
    
    # Argumento por definição
    definition_markers = ['define-se', 'conceitua-se', 'entende-se por', 'significa', 
                         'é caracterizado', 'consiste em', 'trata-se de']
    if any(marker in text_lower for marker in definition_markers):
        arguments_found.append('por-definição')
        _linguistic_memory['argument_styles']['por-definição'] += 1
    
    # Argumento por consequência
    consequence_markers = ['resulta', 'implica', 'consequência', 'acarreta', 
                          'enseja', 'decorre', 'gera', 'provoca']
    if any(marker in text_lower for marker in consequence_markers):
        arguments_found.append('por-consequência')
        _linguistic_memory['argument_styles']['por-consequência'] += 1
    
    # Argumento por comparação
    comparison_markers = ['diferente de', 'ao contrário', 'enquanto', 'já', 
                         'em comparação', 'versus', 'entre', 'ambos']
    if any(marker in text_lower for marker in comparison_markers):
        arguments_found.append('por-comparação')
        _linguistic_memory['argument_styles']['por-comparação'] += 1
    
    # Estatísticas de estilos aprendidos
    total_args = sum(_linguistic_memory['argument_styles'].values())
    style_distribution = {
        style: (count / total_args * 100) if total_args > 0 else 0
        for style, count in _linguistic_memory['argument_styles'].items()
    }
    
    return {
        'arguments_in_text': arguments_found,
        'total_arguments_analyzed': total_args,
        'style_distribution': style_distribution,
        'preferred_style': max(_linguistic_memory['argument_styles'], 
                              key=_linguistic_memory['argument_styles'].get) if _linguistic_memory['argument_styles'] else 'Nenhum',
        'learning_note': 'O agente aprende estratégias persuasivas e adapta seu estilo argumentativo'
    }

def analyze_writing_style(text: str) -> Dict[str, any]:
    """
    APRENDIZADO: Estilos de escrita (características linguísticas).
    
    Analisa características estilísticas:
    - Formalidade (vocabulário técnico vs coloquial)
    - Complexidade (tamanho de frases, estruturas sintáticas)
    - Tom (assertivo, cauteloso, neutro, persuasivo)
    - Densidade técnica (termos especializados por frase)
    - Objetividade (uso de primeira pessoa, hedging)
    
    Args:
        text (str): Texto para análise
        
    Returns:
        Dict com características estilísticas identificadas
    """
    global _linguistic_memory
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    words = text.split()
    
    # 1. Análise de formalidade
    formal_markers = ['solicito', 'requeiro', 'outrossim', 'destarte', 'mister', 
                     'consoante', 'hodierno', 'exordial', 'impende']
    informal_markers = ['tipo', 'né', 'meio', 'tipo assim', 'basicamente']
    
    formal_count = sum(1 for marker in formal_markers if marker in text.lower())
    informal_count = sum(1 for marker in informal_markers if marker in text.lower())
    
    if formal_count > informal_count:
        formality = 'formal-alto'
    elif formal_count > 0:
        formality = 'formal-moderado'
    else:
        formality = 'informal-neutro'
    
    # 2. Análise de complexidade
    if len(sentences) > 0:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        avg_sentence_length = 0
    
    if avg_sentence_length > 25:
        complexity = 'complexo'
    elif avg_sentence_length > 15:
        complexity = 'moderado'
    else:
        complexity = 'simples'
    
    # 3. Análise de tom
    assertive_markers = ['deve', 'é necessário', 'obrigatório', 'imperativo', 'exige']
    cautious_markers = ['pode', 'talvez', 'possivelmente', 'eventualmente', 'sugere-se']
    
    assertive_count = sum(1 for marker in assertive_markers if marker in text.lower())
    cautious_count = sum(1 for marker in cautious_markers if marker in text.lower())
    
    if assertive_count > cautious_count:
        tone = 'assertivo'
    elif cautious_count > assertive_count:
        tone = 'cauteloso'
    else:
        tone = 'neutro'
    
    # 4. Densidade técnica
    technical_terms = ['jurisprudência', 'doutrina', 'acórdão', 'súmula', 'normativo',
                      'edital', 'licitação', 'pregão', 'inexigibilidade', 'dispensa']
    technical_count = sum(1 for term in technical_terms if term in text.lower())
    
    if len(sentences) > 0:
        technical_density = technical_count / len(sentences)
    else:
        technical_density = 0
    
    if technical_density > 0.5:
        density = 'alta-técnica'
    elif technical_density > 0.2:
        density = 'média-técnica'
    else:
        density = 'baixa-técnica'
    
    # 5. Objetividade
    first_person = len(re.findall(r'\b(eu|meu|minha|meus|minhas)\b', text.lower()))
    hedging = len(re.findall(r'\b(acho|penso|parece|aparentemente|provavelmente)\b', text.lower()))
    
    if first_person + hedging > 5:
        objectivity = 'subjetivo'
    elif first_person + hedging > 0:
        objectivity = 'semi-objetivo'
    else:
        objectivity = 'objetivo'
    
    # Registra aprendizado
    style_profile = f"{formality}|{complexity}|{tone}|{density}|{objectivity}"
    _linguistic_memory['writing_styles']['profiles'].append(style_profile)
    
    return {
        'formality': formality,
        'complexity': complexity,
        'tone': tone,
        'technical_density': density,
        'objectivity': objectivity,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'style_profile': style_profile,
        'total_styles_learned': len(_linguistic_memory['writing_styles']['profiles']),
        'learning_note': 'O agente aprende características estilísticas e adapta sua escrita ao contexto'
    }

def get_comprehensive_linguistic_analysis(text: str) -> str:
    """
    Gera relatório completo de análise linguística e aprendizado.
    
    Args:
        text (str): Texto para análise completa
        
    Returns:
        str: Relatório formatado com todas as análises
    """
    try:
        word_rels = analyze_word_relations(text)
    except Exception as e:
        word_rels = {'learned_vocabulary': 0, 'learning_note': f'Erro na análise: {str(e)}'}
    
    try:
        structures = analyze_sentence_structure(text)
    except Exception as e:
        structures = {'total_patterns_learned': 0, 'most_common_structure': 'N/A', 'learning_note': f'Erro: {str(e)}'}
    
    try:
        reasoning = analyze_reasoning_patterns(text)
    except Exception as e:
        reasoning = {'total_reasoning_examples': 0, 'patterns_detected': [], 'learning_note': f'Erro: {str(e)}'}
    
    try:
        arguments = analyze_argument_styles(text)
    except Exception as e:
        arguments = {'total_arguments_analyzed': 0, 'preferred_style': 'N/A', 'arguments_in_text': [], 'learning_note': f'Erro: {str(e)}'}
    
    try:
        style = analyze_writing_style(text)
    except Exception as e:
        style = {'formality': 'N/A', 'complexity': 'N/A', 'tone': 'N/A', 'technical_density': 'N/A', 
                'objectivity': 'N/A', 'avg_sentence_length': 0, 'total_styles_learned': 0, 'learning_note': f'Erro: {str(e)}'}
    
    report = f"""
📚 **ANÁLISE LINGUÍSTICA COMPLETA & APRENDIZADO**

🔗 **RELAÇÕES ENTRE PALAVRAS:**
   • Vocabulário aprendido: {word_rels['learned_vocabulary']} palavras
   • {word_rels['learning_note']}

📝 **ESTRUTURA DE FRASES:**
   • Total de padrões aprendidos: {structures['total_patterns_learned']}
   • Estrutura mais comum: {structures['most_common_structure']}
   • {structures['learning_note']}

🧠 **PADRÕES DE RACIOCÍNIO:**
   • Exemplos analisados: {reasoning['total_reasoning_examples']}
   • Padrões detectados: {', '.join(reasoning['patterns_detected']) if reasoning['patterns_detected'] else 'Nenhum no texto atual'}
   • {reasoning['learning_note']}

💬 **FORMAS DE ARGUMENTAR:**
   • Total de argumentos analisados: {arguments['total_arguments_analyzed']}
   • Estilo preferido: {arguments['preferred_style']}
   • Argumentos no texto: {', '.join(arguments['arguments_in_text']) if arguments['arguments_in_text'] else 'Nenhum'}
   • {arguments['learning_note']}

✍️ **ESTILO DE ESCRITA:**
   • Formalidade: {style['formality']}
   • Complexidade: {style['complexity']} (média {style['avg_sentence_length']} palavras/frase)
   • Tom: {style['tone']}
   • Densidade técnica: {style['technical_density']}
   • Objetividade: {style['objectivity']}
   • Perfis aprendidos: {style['total_styles_learned']}
   • {style['learning_note']}

🎯 **CAPACIDADE DE APRENDIZADO:**
O agente agora "entende" melhor a linguagem através de:
• Identificação de relações semânticas entre palavras
• Reconhecimento de estruturas sintáticas comuns
• Detecção de padrões lógicos de raciocínio
• Análise de estratégias argumentativas
• Adaptação ao estilo de escrita contextual

Cada interação enriquece a base de conhecimento linguístico!
"""
    
    return report

def load_config():
    """Carrega as configurações do agente a partir do arquivo YAML."""
    try:
        # Caminho relativo considerando que o script está em src/
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    except Exception as e:
        print(f"Erro ao carregar config: {e}")
        return {}

def tokenize_and_analyze(text: str) -> Dict[str, any]:
    """
    Simula o processo de tokenização que modelos Transformer fazem.
    
    ⚠️ FUNDAMENTO CRÍTICO: ANTES DE ENTENDER TEXTO, O MODELO QUEBRA TUDO EM TOKENS
    
    TOKEN ≠ PALAVRA NECESSARIAMENTE!
    
    Tokenização é o PRIMEIRO PASSO, antes de qualquer "compreensão":
    1. O modelo recebe texto bruto (string)
    2. Quebra em tokens usando algoritmos como BPE (Byte-Pair Encoding) ou WordPiece
    3. Converte cada token em um número (ID do vocabulário)
    4. Transforma esses números em vetores (embeddings)
    5. SÓ ENTÃO processa com atenção e camadas neurais
    
    Exemplos reais de tokenização (GPT):
    - "licitação" → ["lic", "ita", "ção"] (3 tokens, 1 palavra)
    - "art." → ["art", "."] (2 tokens, 1 palavra)
    - "14.133/2021" → ["14", ".", "133", "/", "2021"] (5 tokens, 1 palavra)
    - "inexigibilidade" → ["inex", "igibil", "idade"] (3 tokens, 1 palavra)
    - "o" → ["o"] (1 token, 1 palavra)
    
    Palavras raras = mais tokens (menos eficiente)
    Palavras comuns = menos tokens (mais eficiente)
    
    Args:
        text (str): Texto para tokenizar
        
    Returns:
        Dict com análise detalhada de tokenização
    """
    # Simulação: em modelos reais, cada palavra é quebrada diferentemente
    words = text.split()
    
    # Aproximação realista: 1 palavra ≈ 1.3 tokens em português
    # (palavras longas viram múltiplos tokens)
    estimated_tokens = 0
    tokenization_examples = []
    
    for word in words[:10]:  # Analisa primeiras 10 palavras como exemplo
        # Simula tokenização: palavras longas = mais tokens
        word_length = len(word)
        
        if word_length <= 3:
            tokens_count = 1
        elif word_length <= 7:
            tokens_count = 2
        else:
            tokens_count = 3
        
        estimated_tokens += tokens_count
        
        # Exemplos de como seria quebrado
        if tokens_count > 1:
            tokenization_examples.append({
                'palavra': word,
                'tokens_estimados': tokens_count,
                'exemplo_quebra': f"[{word[:len(word)//2]}] + [{word[len(word)//2:]}]"
            })
    
    # Estima total de tokens para o texto completo
    avg_tokens_per_word = estimated_tokens / max(len(words[:10]), 1)
    total_estimated_tokens = int(len(words) * avg_tokens_per_word)
    
    # Identifica termos que receberiam alta "atenção" (self-attention)
    high_attention_words = []
    legal_terms = ['lei', 'artigo', 'licitação', 'edital', 'contrato', 'normativa', 
                   'jurisprudência', 'acórdão', 'parecer', 'dispensa', 'inexigibilidade']
    
    for word in words:
        if any(term in word.lower() for term in legal_terms):
            high_attention_words.append(word)
    
    return {
        'word_count': len(words),
        'estimated_tokens': total_estimated_tokens,
        'token_to_word_ratio': round(total_estimated_tokens / max(len(words), 1), 2),
        'tokenization_examples': tokenization_examples[:5],  # Top 5 exemplos
        'high_attention_words': high_attention_words[:10],
        'complexity_score': len(high_attention_words) / max(len(words), 1),
        'processing_note': '⚠️ TOKENS SÃO PROCESSADOS ANTES DE PALAVRAS - O modelo não vê "palavras", vê sequências de IDs numéricos',
        'key_insight': 'Tokenização acontece ANTES de tudo: texto → tokens → IDs → embeddings → processamento'
    }

def extract_attention_keywords(user_query: str, context_texts: List[str]) -> Dict[str, List[str]]:
    """
    Implementa versão simplificada do mecanismo de ATENÇÃO (Self-Attention).
    
    FUNDAMENTO: Quando você faz uma pergunta longa, a IA usa "atenção" para identificar
    quais palavras são mais importantes. Isso permite que ela foque no que realmente importa.
    
    Args:
        user_query (str): Pergunta do usuário
        context_texts (List[str]): Textos de contexto (documentos, RAG, web)
        
    Returns:
        Dict com palavras-chave priorizadas por fonte
    """
    query_analysis = tokenize_and_analyze(user_query)
    
    # Palavras com alta atenção da query
    query_keywords = set(query_analysis['high_attention_words'])
    
    # Busca correspondências nos contextos (simulando attention scores)
    attention_map = {
        'query_focus': list(query_keywords),
        'document_matches': [],
        'rag_matches': [],
        'web_matches': []
    }
    
    # Analisa cada fonte de contexto
    for idx, context in enumerate(context_texts):
        if not context or len(context) < 10:
            continue
            
        # Conta quantas palavras importantes aparecem no contexto
        matches = []
        for keyword in query_keywords:
            if keyword.lower() in context.lower():
                matches.append(keyword)
        
        if idx == 0:  # Documento principal
            attention_map['document_matches'] = matches
        elif idx == 1:  # RAG context
            attention_map['rag_matches'] = matches
        elif idx == 2:  # Web context
            attention_map['web_matches'] = matches
    
    return attention_map

# Memória de curto prazo para contexto conversacional
_conversation_memory = defaultdict(list)

def add_to_conversation_memory(session_id: str, role: str, content: str, max_messages: int = 10):
    """
    Mantém histórico da conversa (contexto imediato / memória de curto prazo).
    
    FUNDAMENTO: A IA usa o que você acabou de escrever para manter o fio da meada.
    Isso fica na "memória de curto prazo" da conversa.
    
    Args:
        session_id (str): ID da sessão de conversa
        role (str): 'user' ou 'assistant'
        content (str): Conteúdo da mensagem
        max_messages (int): Máximo de mensagens a manter (evita overflow de contexto)
    """
    global _conversation_memory
    
    _conversation_memory[session_id].append({
        'role': role,
        'content': content[:500],  # Limita tamanho para economizar memória
        'timestamp': os.times().elapsed
    })
    
    # Mantém apenas as últimas N mensagens
    if len(_conversation_memory[session_id]) > max_messages:
        _conversation_memory[session_id] = _conversation_memory[session_id][-max_messages:]

def get_conversation_context(session_id: str) -> str:
    """
    Recupera o contexto conversacional para manter coerência entre interações.
    
    Returns:
        str: Resumo formatado do histórico da conversa
    """
    global _conversation_memory
    
    if session_id not in _conversation_memory or not _conversation_memory[session_id]:
        return "Primeira interação desta sessão."
    
    history = _conversation_memory[session_id]
    
    context_summary = f"Histórico recente ({len(history)} interações):\n"
    for msg in history[-3:]:  # Últimas 3 mensagens
        role_label = "Usuário" if msg['role'] == 'user' else "Assistente"
        preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
        context_summary += f"- {role_label}: {preview}\n"
    
    return context_summary

def extract_sources_from_context(web_context: str, rag_results: List = None) -> Dict[str, List[str]]:
    """
    Extrai e organiza fontes citáveis do contexto (essencial para RAG).
    
    FUNDAMENTO RAG: Em vez de confiar apenas no que "decorou" no treino,
    o agente busca informações atualizadas e CITA AS FONTES.
    
    Args:
        web_context (str): Contexto da busca web
        rag_results (List): Resultados da busca RAG
        
    Returns:
        Dict com fontes organizadas por tipo
    """
    sources = {
        'web_urls': [],
        'web_titles': [],
        'rag_documents': [],
        'citation_ready': []
    }
    
    # Extrai URLs do contexto web
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, web_context)
    sources['web_urls'] = list(set(urls))[:5]  # Máximo 5 URLs únicas
    
    # Extrai títulos (formato: TITULO: ...)
    title_pattern = r'TITULO:\s*([^\n]+)'
    titles = re.findall(title_pattern, web_context, re.IGNORECASE)
    sources['web_titles'] = titles[:5]
    
    # Processa resultados RAG
    if rag_results:
        for doc in rag_results[:5]:
            source_name = doc.metadata.get('source', 'Documento sem título') if hasattr(doc, 'metadata') else 'RAG Document'
            sources['rag_documents'].append(source_name)
    
    # Cria citações prontas para uso
    for i, (title, url) in enumerate(zip(sources['web_titles'], sources['web_urls']), 1):
        sources['citation_ready'].append(f"[{i}] {title} - {url}")
    
    for doc in sources['rag_documents']:
        sources['citation_ready'].append(f"📄 {doc}")
    
    return sources


def analyze_user_intent(user_query: str) -> Dict[str, any]:
    """
    Analisa a intenção do usuário através de padrões linguísticos.
    
    Args:
        user_query (str): A consulta do usuário
        
    Returns:
        Dict com:
        - intent: tipo de intenção (analisar, informar, redigir, comparar, avaliar)
        - keywords: palavras-chave identificadas
        - complexity: nível de complexidade (simples, médio, complexo)
        - tone: tom esperado (formal, técnico, simplificado)
    """
    query_lower = user_query.lower()
    
    # Padrões de intenção
    intent_patterns = {
        'analisar': ['analis', 'avaliar', 'verificar', 'checar', 'examinar', 'investigar', 'estudar'],
        'informar': ['informar', 'explicar', 'descrever', 'qual', 'quais', 'como', 'quando'],
        'redigir': ['redigir', 'elaborar', 'escrever', 'gerar', 'criar', 'produzir', 'parecer'],
        'comparar': ['comparar', 'diferença', 'versus', 'vs', 'entre', 'melhor', 'pior'],
        'avaliar': ['viável', 'possível', 'legal', 'legalidade', 'regularidade', 'conformidade']
    }
    
    # Identifica intenção principal
    intent_scores = {}
    for intent, patterns in intent_patterns.items():
        score = sum(1 for pattern in patterns if pattern in query_lower)
        if score > 0:
            intent_scores[intent] = score
    
    primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'informar'
    
    # Extrai palavras-chave (substantivos e termos técnicos)
    keywords = []
    technical_terms = ['edital', 'licitação', 'contrato', 'dispensa', 'inexigibilidade', 
                      'pregão', 'concorrência', 'tomada de preços', 'convite',
                      'lei 14.133', 'lei 8.666', 'tcu', 'stj', 'stf']
    
    for term in technical_terms:
        if term in query_lower:
            keywords.append(term)
    
    # Determina complexidade baseado no tamanho e estrutura da query
    word_count = len(user_query.split())
    has_multiple_clauses = len(re.findall(r'[,;.]', user_query)) > 2
    
    if word_count < 10 and not has_multiple_clauses:
        complexity = 'simples'
    elif word_count < 30:
        complexity = 'médio'
    else:
        complexity = 'complexo'
    
    # Determina tom baseado em palavras formais/técnicas
    formal_words = ['solicito', 'requeiro', 'peço', 'necessário', 'favor']
    technical_words = ['normativo', 'jurisprudência', 'doutrina', 'acórdão', 'súmula']
    
    has_formal = any(word in query_lower for word in formal_words)
    has_technical = any(word in query_lower for word in technical_words)
    
    if has_technical:
        tone = 'técnico'
    elif has_formal:
        tone = 'formal'
    else:
        tone = 'simplificado'
    
    return {
        'intent': primary_intent,
        'keywords': keywords,
        'complexity': complexity,
        'tone': tone,
        'word_count': word_count
    }

def generate_legal_opinion(
    pdf_text: str, 
    rag_context: str, 
    web_context: str, 
    user_query: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    rag_results: Optional[List] = None,
    session_id: Optional[str] = "default",
    template_text: Optional[str] = None,
    style_examples: Optional[str] = None,
    output_mode: Optional[str] = None,
    chapter: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Gera um parecer jurídico com base no texto do PDF, contexto do RAG, busca na web e consulta do usuário.
    
    Implementa princípios fundamentais de IA:
    - Tokenização e análise de atenção (Transformer)
    - Previsão estatística baseada em contexto
    - RAG (Retrieval-Augmented Generation) com fontes
    - Memória de conversação (contexto imediato)
    
    Args:
        pdf_text: Texto extraído de documentos
        rag_context: Contexto da base de conhecimento local
        web_context: Contexto de busca web
        user_query: Pergunta/solicitação do usuário
        model_name: Modelo LLM a usar
        temperature: Criatividade da resposta (0-1)
        rag_results: Lista de documentos RAG (opcional, para citação)
        session_id: ID da sessão para manter contexto conversacional
        template_text: Template base para orientar a estrutura
        output_mode: Modo de saída (técnico, simplificado, minuta)
        chapter: Capítulo específico a gerar (relatorio, fundamentacao, conclusao)
    """
    
    # 1. TOKENIZAÇÃO E ANÁLISE DE ATENÇÃO (Transformer)
    # Fundamento: IA processa tokens, não palavras completas
    token_analysis = tokenize_and_analyze(user_query)
    
    # 2. MECANISMO DE ATENÇÃO (Self-Attention)
    # Fundamento: Identifica quais palavras são mais importantes
    attention_map = extract_attention_keywords(
        user_query, 
        [pdf_text, rag_context, web_context]
    )
    
    # 3. MEMÓRIA DE CURTO PRAZO (Contexto Conversacional)
    # Fundamento: Mantém o fio da meada da conversa
    add_to_conversation_memory(session_id, 'user', user_query)
    conversation_context = get_conversation_context(session_id)
    
    # 4. RAG - EXTRAÇÃO DE FONTES
    # Fundamento: Não apenas "decorar", mas BUSCAR e CITAR fontes
    sources = extract_sources_from_context(web_context, rag_results)
    
    # 5. ANÁLISE LINGUÍSTICA PROFUNDA E APRENDIZADO
    # Fundamento: O agente "aprende" padrões linguísticos em tempo real
    linguistic_analysis = {
        'word_relations': analyze_word_relations(user_query + ' ' + pdf_text[:1000]),
        'sentence_structures': analyze_sentence_structure(user_query),
        'reasoning_patterns': analyze_reasoning_patterns(pdf_text[:2000]),
        'argument_styles': analyze_argument_styles(pdf_text[:2000]),
        'writing_style': analyze_writing_style(pdf_text[:2000])
    }
    
    # 6. Analisar intenção do usuário
    user_analysis = analyze_user_intent(user_query)

    # 6.1 Classificar o tipo de caso e carregar o template padrão
    if not template_text:
        case_type = classify_legal_case(user_query)
        template_text = load_template(case_type)
    
    # 7. Carregar Configurações
    config = load_config()
    # Se falhar o load, usa defaults vazios para não quebrar
    agent_config = config.get("agent", {}) if config else {}
    
    name = agent_config.get("name", "Agente Jurídico")
    domains = ", ".join(agent_config.get("domain", ["Direito Administrativo"]))
    
    secrecy = agent_config.get("secrecy", {})
    placeholders = ", ".join(secrecy.get("placeholders", ["[NOME]", "[DADO]"]))
    redact_fields = ", ".join(secrecy.get("redact_fields", []))
    
    output_cfg = agent_config.get("output", {})
    structure = "\n       ".join(output_cfg.get("default_structure", [
        "- Relatório", "- Fundamentação", "- Conclusão"
    ]))
    formatting = ", ".join(output_cfg.get("formatting", {}).get("bold_for", ["Prazos"]))

    quality_gates = agent_config.get("quality_gates", {}).get("must_pass", [])
    gates_str = ""
    for g in quality_gates:
        gates_str += f"- {g.get('name')}: {g.get('rule')}\n       "
        
    triggers = ", ".join(agent_config.get("escalation", {}).get("triggers", []))

    # Configuração de Regeneração (Correção Automática)
    regen_config = config.get("regen", {})
    regen_instruction = ""
    if regen_config.get("enabled"):
        updates = ", ".join(regen_config.get("post_delivery_updates", []))
        regen_instruction = f"""
    6. REGENERAÇÃO E AUTO-CORREÇÃO (Loop de Aprendizado):
       - ATENÇÃO: Se esta for uma solicitação de correção (feedback), trate como uma NOVA VERSÃO (v+1).
       - Aplique as seguintes ações de melhoria: {updates}.
       - Ao final, liste explicitamente o que foi corrigido em relação à versão anterior.
        """

    # 8. Adaptar comportamento baseado na análise linguística e de intenção
    
    # Instrução de adaptação de estilo baseada no aprendizado
    style_instruction = f"""
    🎨 ADAPTAÇÃO DE ESTILO (Aprendizado Linguístico):
    Baseado na análise do texto fornecido, adapte sua resposta para:
    - Formalidade: {linguistic_analysis['writing_style']['formality']}
    - Complexidade: {linguistic_analysis['writing_style']['complexity']}
    - Tom: {linguistic_analysis['writing_style']['tone']}
    - Densidade técnica: {linguistic_analysis['writing_style']['technical_density']}
    
    Reproduza estruturas de frases semelhantes às identificadas:
    - Padrão dominante: {linguistic_analysis['sentence_structures'].get('most_common_structure', 'variado')}
    
    Use estratégias argumentativas aprendidas:
    - Estilo preferido: {linguistic_analysis['argument_styles'].get('preferred_style', 'misto')}
    """
    
    # 9. Adaptar comportamento baseado na análise de intenção
    intent_instructions = ""
    if user_analysis['intent'] == 'analisar':
        intent_instructions = """
    FOCO DA RESPOSTA: ANÁLISE CRÍTICA
    - Identifique pontos fortes e fracos do documento
    - Aponte riscos jurídicos específicos
    - Sugira melhorias ou correções necessárias
    - Use estrutura comparativa (o que está / o que deveria estar)
        """
    elif user_analysis['intent'] == 'informar':
        intent_instructions = """
    FOCO DA RESPOSTA: INFORMAÇÃO CLARA E OBJETIVA
    - Responda diretamente à pergunta
    - Use linguagem clara e acessível
    - Priorize exemplos práticos
    - Evite digressões desnecessárias
        """
    elif user_analysis['intent'] == 'redigir':
        intent_instructions = """
    FOCO DA RESPOSTA: REDAÇÃO COMPLETA E ESTRUTURADA
    - Produza texto formal e bem estruturado
    - Inclua todos os elementos obrigatórios
    - Use formatação profissional
    - Forneça texto pronto para uso
        """
    elif user_analysis['intent'] == 'comparar':
        intent_instructions = """
    FOCO DA RESPOSTA: COMPARAÇÃO SISTEMÁTICA
    - Use tabelas ou listas para comparar
    - Destaque diferenças e semelhanças
    - Indique vantagens e desvantagens de cada opção
    - Conclua com recomendação justificada
        """
    elif user_analysis['intent'] == 'avaliar':
        intent_instructions = """
    FOCO DA RESPOSTA: AVALIAÇÃO TÉCNICA
    - Emita opinião fundamentada sobre viabilidade
    - Liste requisitos cumpridos e não cumpridos
    - Quantifique riscos (baixo/médio/alto)
    - Forneça parecer conclusivo claro
        """
    
    # 4. Adaptar nível de detalhe baseado na complexidade
    detail_level = ""
    if user_analysis['complexity'] == 'simples':
        detail_level = """
    NÍVEL DE DETALHE: OBJETIVO E DIRETO
    - Resposta concisa (máximo 2-3 parágrafos por seção)
    - Vá direto ao ponto
    - Evite citações extensas
    - Priorize conclusões práticas
        """
    elif user_analysis['complexity'] == 'médio':
        detail_level = """
    NÍVEL DE DETALHE: EQUILIBRADO
    - Resposta moderada (4-6 parágrafos por seção)
    - Balance fundamentação e objetividade
    - Inclua citações relevantes
    - Explique conceitos-chave
        """
    else:  # complexo
        detail_level = """
    NÍVEL DE DETALHE: APROFUNDADO E TÉCNICO
    - Resposta completa e exaustiva
    - Análise detalhada de todos os aspectos
    - Inclua jurisprudência e doutrina extensiva
    - Explore nuances e interpretações divergentes
        """
    
    # 5. Adaptar tom da linguagem
    tone_style = ""
    if user_analysis['tone'] == 'técnico':
        tone_style = """
    TOM DA LINGUAGEM: TÉCNICO-JURÍDICO
    - Use terminologia jurídica precisa
    - Cite artigos de lei com número completo
    - Referencie jurisprudência e doutrina
    - Mantenha formalidade acadêmica
        """
    elif user_analysis['tone'] == 'formal':
        tone_style = """
    TOM DA LINGUAGEM: FORMAL PROFISSIONAL
    - Use linguagem formal mas acessível
    - Evite jargão excessivo
    - Explique termos técnicos quando necessário
    - Mantenha respeito sem pedantismo
        """
    else:  # simplificado
        tone_style = """
    TOM DA LINGUAGEM: CLARO E ACESSÍVEL
    - Use linguagem simples e direta
    - Traduza termos técnicos para linguagem comum
    - Use analogias quando apropriado
    - Priorize comunicação efetiva sobre formalidade
        """
    
    # 6. Instrução sobre palavras-chave identificadas
    keywords_instruction = ""
    if user_analysis['keywords']:
        keywords_instruction = f"""
    FOCO TEMÁTICO IDENTIFICADO:
    O usuário mencionou especificamente: {', '.join(user_analysis['keywords'])}
    - Priorize esses tópicos na sua resposta
    - Certifique-se de abordar cada termo mencionado
    - Use esses termos como guia para estruturar a resposta
        """

    # Configuração do modelo com max_tokens dentro dos limites reais de conclusão
    # Observação: modelos atuais suportam ~4k tokens de saída; usar valores maiores gera erro 400.
    max_tokens_config = {
        "gpt-4-turbo": 4000,
        "gpt-4-turbo-preview": 4000,
        "gpt-4o": 4000,
        "gpt-4o-mini": 4000,
        "gpt-4": 4000,
        "gpt-3.5-turbo": 4000
    }
    
    max_tokens = max_tokens_config.get(model_name, 4096)
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True  # Garante streaming para respostas longas
    )
    
    # Preparar informações sobre fontes para incluir no prompt
    sources_instruction = ""
    if sources['citation_ready']:
        sources_list = "\n       ".join(sources['citation_ready'][:10])
        sources_instruction = f"""
    FONTES DISPONÍVEIS PARA CITAÇÃO:
    {sources_list}
    
    IMPORTANTE: Cite estas fontes sempre que usar informações delas.
    Formato de citação: "Conforme [fonte], ..." ou "De acordo com [documento], ..."
        """

    now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    system_prompt = f"""
    IDENTITY:
    Você é o "{name}", especialista em {domains}.
    Sua missão é atuar como uma barreira de controle de legalidade (Compliance Check).
    
    FUNDAMENTOS DE FUNCIONAMENTO (Como esta IA opera):
    
    🧠 ARQUITETURA TRANSFORMER E TOKENS:
    ⚠️ ANTES DE ENTENDER TEXTO, O MODELO QUEBRA TUDO EM TOKENS
    
    PROCESSO DE TOKENIZAÇÃO (ETAPA 1 - PRÉ-PROCESSAMENTO):
    1. Texto bruto → Algoritmo BPE/WordPiece → Tokens (pedaços)
    2. Tokens → IDs numéricos (vocabulário)
    3. IDs → Embeddings (vetores de 768-4096 dimensões)
    4. SÓ ENTÃO: Processamento neural (atenção, camadas)
    
    IMPORTANTE: TOKEN ≠ PALAVRA NECESSARIAMENTE
    - "licitação" pode virar 3 tokens: ["lic", "ita", "ção"]
    - "14.133/2021" vira ~5 tokens: ["14", ".", "133", "/", "2021"]
    - Palavras raras = mais tokens (menos eficiente)
    - Espaços e pontuação também são tokens
    
    ANÁLISE DA QUERY ATUAL:
    - Palavras na query: {token_analysis['word_count']}
    - Tokens estimados: {token_analysis['estimated_tokens']} tokens
    - Razão token/palavra: {token_analysis.get('token_to_word_ratio', 1.3)}:1
    - Termos de alta atenção: {', '.join(token_analysis.get('high_attention_words', [])[:5])}
    - Insight: {token_analysis.get('key_insight', 'Tokenização é a base de tudo')}
    
    👁️ MECANISMO DE ATENÇÃO (Self-Attention):
    - Você identifica automaticamente as palavras mais importantes da pergunta
    - Palavras-foco detectadas: {', '.join(attention_map['query_focus'][:5])}
    - Correspondências no documento: {len(attention_map.get('document_matches', []))} termos
    - Correspondências no RAG: {len(attention_map.get('rag_matches', []))} termos
    - Correspondências na web: {len(attention_map.get('web_matches', []))} termos
    
    📊 PREVISÃO ESTATÍSTICA:
    - Você calcula PROBABILIDADES das próximas palavras baseado em bilhões de exemplos
    - Não "sabe" fatos absolutos - você prevê padrões linguísticos mais prováveis
    - Por isso é ESSENCIAL citar fontes externas quando disponíveis (RAG)
    
    💭 CONTEXTO CONVERSACIONAL (Memória de Curto Prazo):
    {conversation_context}
    - Use este histórico para manter coerência e continuidade
    
    🔍 RAG - GERAÇÃO AUMENTADA POR RECUPERAÇÃO:
    Em vez de confiar apenas no treinamento, você deve:
    1. BUSCAR informações nas fontes fornecidas
    2. LER e PROCESSAR o conteúdo encontrado
    3. RESUMIR com CITAÇÃO OBRIGATÓRIA das fontes
    {sources_instruction}
    
    📚 APRENDIZADO LINGUÍSTICO CONTÍNUO:
    Você aprende padrões linguísticos a cada interação:
    - Relações entre palavras: {linguistic_analysis['word_relations']['learned_vocabulary']} termos conectados
    - Estruturas de frases aprendidas: {linguistic_analysis['sentence_structures']['total_patterns_learned']} padrões
    - Padrões de raciocínio: {linguistic_analysis['reasoning_patterns']['total_reasoning_examples']} exemplos
    - Estilos argumentativos: {linguistic_analysis['argument_styles']['total_arguments_analyzed']} argumentos
    
    {style_instruction}
    
    🌐 INTEGRAÇÃO COM FONTES EXTERNAS (Prioridade: Bibliotecário Ágil):
    PRINCÍPIO FUNDAMENTAL: Priorize SEMPRE fontes externas sobre "conhecimento interno"
    
    Hierarquia de confiabilidade:
    1. 🏛️ Legislação oficial (portais do governo, Planalto, STF, STJ, TCU)
    2. 📊 Jurisprudência e precedentes (tribunais, súmulas)
    3. 📄 Documentos do processo (RAG - base local)
    4. 🔍 Busca web atual (consultas em tempo real)
    5. 🧠 Conhecimento de treinamento (última opção, sempre com ressalvas)
    
    Como agir como "bibliotecário ágil":
    - Busque ativamente por fontes atualizadas
    - Cite SEMPRE a origem da informação
    - Indique quando dados estão desatualizados ou faltando
    - Cruze informações de múltiplas fontes (vídeos, artigos, mapas, documentos)
    - Seja transparente sobre limitações e lacunas
    
    ACESSO À INTERNET E CRUZAMENTO DE DADOS:
    - Busca web fornecida: disponível no contexto web_context
    - Use para verificar legislação atualizada, jurisprudência recente
    - Cruze dados: se mencionar prazos, verifique na web; se citar leis, busque texto atualizado
    - Integre informações temporais (data atual: {now_str})
    
    💬 FLUIDEZ E LÓGICA CONVERSACIONAL:
    Você deve ser um assistente versátil, não apenas um gerador de pareceres:
    - Mantenha COERÊNCIA com mensagens anteriores da sessão
    - Adapte-se ao estilo do usuário (formal/informal, técnico/leigo)
    - Faça TRANSIÇÕES suaves entre tópicos
    - Use linguagem NATURAL e fluida, não robótica
    - Demonstre que "entendeu" o contexto geral da conversa
    
    Estrutura conversacional aprendida:
    - Estrutura dominante: {linguistic_analysis['sentence_structures'].get('most_common_structure', 'variado')}
    - Estilo argumentativo preferido: {linguistic_analysis['argument_styles'].get('preferred_style', 'misto')}
    - Tom identificado: {linguistic_analysis['writing_style']['tone']}
    - Complexidade: {linguistic_analysis['writing_style']['complexity']}
    
    ADAPTAÇÃO CONTEXTUAL (Baseado na Análise do Usuário):
    {intent_instructions}
    {detail_level}
    {tone_style}
    {keywords_instruction}

    DIRETRIZES DE SEGURANÇA E SIGILO:
    SIGILO OBRIGATÓRIO: Substitua dados pessoais por {placeholders}. 
    Campos sensíveis: {redact_fields}.
    
    PIPELINE DE EXECUÇÃO OBRIGATÓRIA (Workflow):
    
    1. CLASSIFICAÇÃO IMEDIATA:
       Identifique se o caso é: Edital, Contrato, Convênio ou Contencioso.
       
    2. DIAGNÓSTICO E QUALITY GATES (Checklist de Bloqueio):
       Antes de opinar, verifique se o caso cumpre os requisitos mínimos:
       {gates_str}
       SE ALGUM REQUISITO FALHAR GRAVEMENTE (Ex: {triggers}), ESCALONE PARA VALIDAÇÃO HUMANA.

    3. PROVAS E TESES (Evidence-Based Law):
       - Cruzamento obrigatório: Fato (Página X) -> Norma -> Prova.
       - Se não houver prova documental, declare "INEXISTÊNCIA DE PROVA".

    4. REDAÇÃO FINAL (Estrutura Obrigatória):
       Gere o parecer seguindo rigorosamente esta estrutura:
       {structure}
       
       REGRA ESPECIAL PARA AUSÊNCIA DE DOCUMENTOS:
       Se a FONTE PRIMÁRIA (Documentos Internos) estiver vazia ou não contiver documentos essenciais (edital, contrato, etc.), os tópicos 1, 2, 3, 4, 5, 6, 7 e 8 DEVEM ser EXATAMENTE os seguintes textos:
       
       "1. Sumário Executivo
       Foi solicitada análise de documentação anexada e da biblioteca do projeto, com eventual complementação por pesquisa externa. Após varredura, não foram localizados documentos essenciais (ex.: edital, Termo de Referência, ETP, pesquisa de preços, minuta/contrato, atas, parecer jurídico, publicações, planilhas, mapa de rotas, etc.), o que impede conclusão técnica sobre regularidade, proporcionalidade das exigências, estimativa de preços e viabilidade do objeto.
       
       Conclusão: a análise não é possível em profundidade por lacuna documental. Recomenda-se regularização imediata do acervo e reprocessamento do parecer após juntada dos documentos mínimos, com rastreabilidade por arquivo/página;"
       
       "2. Escopo, Método e Linha do Tempo (auditável)
       
       2.1 Escopo
       Licitações/contratos/convênios: checagem de conformidade documental mínima, riscos de impugnação, inexequibilidade, nulidade, e risco de controle externo.
       
       2.2 Método aplicado e Log de Busca Interna
       - Palavras-chave utilizadas: [LISTAR PALAVRAS-CHAVE USADAS NA BUSCA, ex: "edital", "TR", "contrato"]
       - Locais de busca: Anexos fornecidos (Fonte Primária) e Biblioteca do projeto (RAG)
       - Arquivos encontrados: 0 (zero) documentos essenciais localizados.
       Busca externa (internet): consulta a referências normativas gerais e orientações públicas (sem substituir documentos do processo).
       
       2.3 Linha do tempo
       Data da solicitação: [DADO]
       Prazo para entrega/decisão: [DADO]
       Constatação central: ausência de acervo mínimo para instrução e análise."
       
       "3. Diagnóstico por Quality Gates (com ressalvas)
       
       Motivação: não verificável (sem ETP/TR/justificativa juntada).
       Proporcionalidade: não verificável (sem edital/TR e sem exigências transcritas).
       Documentação: insuficiente (lacuna de peças essenciais).
       Rastreabilidade: inexistente no parecer (não há referência a arquivos/páginas; não há índice de anexos).
       Coerência: não verificável (sem descrição técnica do objeto e critério de julgamento)."
       
       "4. Fundamentação Normativa (base mínima, sem presumir regime do certame)
       
       Sem acesso aos documentos, não é possível afirmar se o procedimento tramita sob Lei nº 14.133/2021 ou sob regime anterior/transitório. Ainda assim, há regra geral: processos licitatórios/contratuais exigem instrução formal e motivada, com documentação mínima que suporte necessidade, preço estimado, regras do certame e gestão do contrato.
       
       - Lei nº 14.133/2021 (Normas gerais de licitações e contratos).
       - Lei nº 8.666/1993 (regime anterior, aplicável conforme transição e contexto do ente).
       
       Ajuste técnico importante: há risco de invalidação/anulação por deficiência de instrução."
       
       "5. Teses aplicáveis (por cenário)
       
       Cenário A — Lacuna de documentação essencial (Lacuna Real)
       Tese técnica: Sem instrução mínima (ETP/TR, edital/minuta, estimativa de preços e anexos do objeto), o processo fica vulnerável a questionamentos de legalidade, nulidade, impugnações e apontamentos de controle externo, porque não se consegue demonstrar:
       - necessidade e adequação do objeto,
       - critério de julgamento e regras do certame,
       - estimativa e aceitabilidade de preços,
       - fiscalização/gestão contratual.
       
       Cenário B — Falha do sistema de anexos/biblioteca (Lacuna de acesso/indexação)
       Tese técnica: Pode haver documentação existente, mas inacessível ao agente por erro de indexação/nomeação/permissão/pasta. Nesse caso, o problema é de governança informacional, e a ação é saneamento do repositório."
       
       "6. Análise de Risco e Evidências
       
       Risco Alto (se a lacuna for real):
       - Controle externo: glosa, determinações, recomendações, responsabilização por falha de instrução.
       - Impugnação e judicialização: ausência de TR/edital/planilhas aumenta probabilidade de suspensão.
       - Inexequibilidade/sobrepreço: sem pesquisa/memória de cálculo, risco de contratar mal.
       - Execução e fiscalização: sem definição técnica, risco de prestação inadequada e litígios.
       
       Risco Moderado (se a lacuna for de acesso/indexação):
       - Atraso e retrabalho; decisões baseadas em informação incompleta."
       
       "7. Pedidos / Recomendações (objetivas e executáveis)
       
       7.1 Regularização do acervo (Lista mínima de documentos por tipo de caso)
       Solicitar/Anexar, no mínimo, conforme o caso:
       
       Para Editais/Licitações:
       - Termo de Referência e/ou ETP (ou justificativa formal equivalente).
       - Edital (ou minuta do instrumento convocatório) e anexos.
       - Pesquisa de preços + memória de cálculo (metodologia e fontes).
       - Parecer jurídico (ou manifestação jurídica), quando aplicável.
       
       Para Contratos/Aditivos:
       - Minuta contratual/Contrato assinado.
       - Termos Aditivos (se existirem) e suas justificativas.
       - Documentos de habilitação e regularidade fiscal da contratada.
       - Publicações (aviso, extrato) e atas.
       
       Para Convênios:
       - Plano de Trabalho aprovado.
       - Termo de Convênio assinado.
       - Comprovação de regularidade do convenente.
       - Pareceres técnicos e jurídicos de aprovação.
       
       7.2 Governança do repositório (para o agente achar)
       - Padronizar nomes: A-001_TR.pdf, A-002_Edital.pdf, A-003_ETP.pdf, A-004_PesquisaPrecos.xlsx, etc.
       - Gerar um index.yaml/index.csv com arquivo + assunto + data + páginas-chave.
       
       7.3 Reprocessamento
       - Após juntada, reexecutar o parecer para produzir: Matriz de evidências, índice de anexos e checklist do edital/TR com pontos de risco."
       
       "8. Apêndice — Matriz de Evidências e Índice de Anexos (negativos, para auditoria)
       
       8.1 Matriz de evidências (NEGATIVA — “deveria existir”)
       | Item a verificar      | Evidência esperada            | Status     |
       | --------------------- | ----------------------------- | ---------- |
       | Necessidade/motivação | ETP/TR/justificativa          | **LACUNA** |
       | Definição do objeto   | TR + anexos técnicos          | **LACUNA** |
       | Estimativa de preços  | pesquisa + memória de cálculo | **LACUNA** |
       | Regras do certame     | edital + anexos               | **LACUNA** |
       | Gestão/fiscalização   | cláusulas e rotinas           | **LACUNA** |
       | Publicidade/atos      | publicações/atas              | **LACUNA** |
       | Parecer jurídico      | peça de controle              | **LACUNA** |
       
       8.2 Índice de Anexos (Vazio)
       - Nenhum anexo essencial foi identificado ou processado com sucesso para esta análise."
       
    5. APRENDIZADO CONTÍNUO:
       Use NEGRITO para: {formatting}.

    {regen_instruction}
    
    ⚠️ **INSTRUÇÃO CRÍTICA DE GERAÇÃO COMPLETA** ⚠️
    
    Você DEVE gerar o parecer COMPLETO em uma única execução.
    NÃO interrompa a geração no meio.
    NÃO use marcadores como "▌" ou reticências "..." no final.
    Se necessário, seja mais conciso em cada seção, mas COMPLETE todas.
    
    **ESTRUTURA OBRIGATÓRIA - TODAS as seções devem ser completadas:**
    
    ✅ 1. Sumário Executivo (obrigatório - mínimo 2 parágrafos)
    ✅ 2. Fatos e Linha do Tempo (obrigatório - com datas)
    ✅ 3. Fundamentação Normativa (obrigatório - com artigos)
    ✅ 4. Teses Aplicáveis (obrigatório - cenários A, B, C se aplicável)
    ✅ 5. Análise de Riscos (obrigatório - com evidências)
    ✅ 6. Pedidos/Recomendações (obrigatório - lista numerada)
    ✅ 7. Matriz de Evidências (obrigatório - tabela)
    ✅ 8. Índice de Anexos (obrigatório - lista)
    ✅ 9. Conclusão (obrigatório - 2-3 parágrafos finais)
    ✅ Citações (obrigatório - todas as fontes [1], [2], [3]...)
    
    📋 **33 PONTOS TÉCNICOS CRÍTICOS - ANÁLISE OBRIGATÓRIA DE TR/EDITAL:**
    
    Ao analisar Termo de Referência, Edital ou processo licitatório, você DEVE verificar e comentar:
    
    **A) NATUREZA DO OBJETO (5 pontos):**
    1. Classificação está correta? (serviço comum/comum continuado/engenharia)
    2. Há ETP (Estudo Técnico Preliminar) formalmente aprovado?
    3. Definição atende art. 6º, XIII e XXIII da Lei 14.133/2021?
    4. Há matriz de riscos mencionada ou anexada?
    5. Objeto é divisível? Justifica divisão por item?
    
    **B) CRITÉRIO DE JULGAMENTO E PREÇOS (6 pontos):**
    6. Critério adotado (menor preço/melhor técnica/maior desconto)?
    7. Divisão por item/lote tem justificativa técnica?
    8. Há memória de cálculo do preço estimado?
    9. Pesquisa de preços conforme art. 23 da Lei 14.133?
    10. Valor total estimado está lastreado em estudo formal?
    11. Planilhas de custos estão detalhadas e realistas?
    
    **C) EXIGÊNCIAS POTENCIALMENTE RESTRITIVAS (8 pontos):**
    12. Há vedação a consórcio? Foi justificada (art. 15)?
    13. Exige veículo já disponível na licitação? (TCU: pode restringir)
    14. Limite de itinerários por empresa? Afronta competitividade?
    15. Capacidade técnica mínima está proporcional (art. 67)?
    16. Percentual de capacidade técnica (ex: 30%) é razoável?
    17. Exigências de qualificação técnica são essenciais?
    18. Há comprovação prévia de posse de bens? Viola jurisprudência?
    19. Critérios de habilitação são objetivos e mensuráveis?
    
    **D) CLÁUSULAS DE RISCO JURÍDICO (7 pontos):**
    20. Índice de reajuste está definido (IPCA/IGP-M)?
    21. Reajuste é coerente com contrato continuado?
    22. Há cláusula clara de reequilíbrio econômico-financeiro?
    23. Revisão por aumento de diesel/combustível está prevista?
    24. Critério de recomposição de preços está tecnicamente definido?
    25. Limite de alteração contratual (ex: 25%) está alinhado ao art. 125?
    26. Prazos de pagamento são compatíveis com fluxo de caixa?
    
    **E) PROPORCIONALIDADE E DETALHAMENTO (4 pontos):**
    27. Há excesso de detalhamento operacional? (risco: engessar fiscalização)
    28. Normas comportamentais são proporcionais?
    29. Obrigações contratuais podem gerar nulidade por extrapolação?
    30. Cláusulas penais são razoáveis e graduadas?
    
    **F) COMPETÊNCIA E RECEPÇÃO NORMATIVA (3 pontos):**
    31. Se município usa norma estadual: há ato formal de recepção?
    32. Adoção de norma externa é válida sob pacto federativo?
    33. Há fundamento legal municipal para não ter lei própria?
    
    ⚠️ **PARA CADA PONTO NÃO VERIFICADO/AUSENTE, MENCIONE EXPLICITAMENTE NO PARECER.**
    
    Exemplo: "⚠️ Ponto 2 (ETP): NÃO LOCALIZADO - risco de anulação por falta de motivação adequada."
    
    Esta checklist DEVE aparecer ao final da seção de Análise de Riscos ou em seção específica.
    """
    
    template_block = template_text if template_text else "(nenhum template fornecido)"
    style_block = style_examples if style_examples else "(nenhum exemplo fornecido)"
    output_mode_label = output_mode if output_mode else "padrao"
    chapter_label = chapter if chapter else "completo"

    rag_limit_chars = 12000
    rag_context_limited = rag_context[:rag_limit_chars]

    user_prompt = f"""
    Use a estrutura jurídica abaixo como base obrigatória para o parecer:

    {template_block}

    Documentos analisados:
    {{pdf_text}}

    Contexto da biblioteca:
    {{rag_context}}

    Jurisprudência e doutrina da web:
    {{web_context}}

    MODELO DE ESTILO E ESTRUTURA (use como guia):
    {style_block}

    Tema da análise:
    {{user_query}}

    Elabore parecer jurídico completo.

    Regras de citacao obrigatoria:
    - Sempre que usar a web, cite URL, dominio e data/hora da coleta.
    - Sempre que usar a biblioteca local, cite a fonte (arquivo).
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    # Limita o tamanho do texto para evitar estourar o contexto do modelo
    # Modelos gpt-4-turbo/gpt-4o suportam 128k tokens de contexto (~512k chars)
    # Usando valor generoso para permitir análises completas
    max_chars_config = {
        "gpt-4-turbo": 200000,      # ~50k tokens - pareceres muito extensos
        "gpt-4-turbo-preview": 200000,
        "gpt-4o": 200000,           # ~50k tokens - pareceres extensos
        "gpt-4o-mini": 150000,      # ~37.5k tokens - pareceres médios
        "gpt-4": 100000,            # ~25k tokens
        "gpt-3.5-turbo": 50000      # ~12.5k tokens - conservador
    }
    
    max_chars = max_chars_config.get(model_name, 50000)
    
    # ===== STREAMING ROBUSTO COM RETRY LOGIC E TIMEOUT =====
    # Sistema de retry para garantir geração completa mesmo com falhas transitórias
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Se o LLM suportar streaming, retornamos o iterador de chunks
            if hasattr(llm, "stream"):
                chain = prompt | llm
                
                # Configurar timeout no cliente OpenAI (5 minutos para pareceres longos)
                # Nota: LangChain usa timeout como parâmetro de configuração
                stream_response = chain.stream(
                    {
                        "pdf_text": pdf_text[:max_chars],
                        "rag_context": rag_context_limited,  # limite de contexto RAG
                        "web_context": web_context[:50000],   # ~12k tokens para Web
                        "user_query": user_query
                    },
                    config={"timeout": 300.0}  # 5 minutos
                )
                
                # Buffer para acumular chunks e otimizar yields
                buffer = ""
                for chunk in stream_response:
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    buffer += content
                    
                    # Yield em lotes de 50 caracteres para reduzir overhead
                    if len(buffer) >= 50:
                        yield buffer
                        buffer = ""
                
                # Yield do buffer restante
                if buffer:
                    yield buffer
                
                # Sucesso - sai do loop de retry
                break
                
            else:
                # Fallback para invoke (sem streaming)
                chain = prompt | llm
                response = chain.invoke({
                    "pdf_text": pdf_text[:max_chars],
                    "rag_context": rag_context[:100000],
                    "web_context": web_context[:50000],
                    "user_query": user_query
                })
                yield response.content
                break
                
        except Exception as e:
            retry_count += 1
            
            if retry_count >= max_retries:
                # Falhou após 3 tentativas - informa o erro
                error_msg = f"\n\n⚠️ **ERRO APÓS {max_retries} TENTATIVAS**\n\n"
                error_msg += f"Erro técnico: {str(e)}\n\n"
                error_msg += "**Sugestões:**\n"
                error_msg += "1. Tente novamente em alguns segundos\n"
                error_msg += "2. Use modelo menor (gpt-4o-mini)\n"
                error_msg += "3. Reduza o tamanho do documento\n"
                error_msg += "4. Verifique sua conexão com a internet\n"
                yield error_msg
                break
            else:
                # Aguarda 2 segundos antes de tentar novamente
                time.sleep(2)
                continue

def get_contextual_analysis_report(user_query: str) -> str:
    """
    Gera um relatório legível da análise contextual da query do usuário.
    Útil para mostrar ao usuário como o agente interpretou a solicitação.
    
    Args:
        user_query (str): A consulta do usuário
        
    Returns:
        str: Relatório formatado da análise contextual
    """
    analysis = analyze_user_intent(user_query)
    
    intent_emoji = {
        'analisar': '🔍',
        'informar': '📋',
        'redigir': '✍️',
        'comparar': '⚖️',
        'avaliar': '🎯'
    }
    
    complexity_emoji = {
        'simples': '🟢',
        'médio': '🟡',
        'complexo': '🔴'
    }
    
    tone_emoji = {
        'técnico': '🎓',
        'formal': '💼',
        'simplificado': '💬'
    }
    
    report = f"""
📊 **ANÁLISE CONTEXTUAL DA SOLICITAÇÃO**

{intent_emoji.get(analysis['intent'], '📌')} **Intenção Detectada:** {analysis['intent'].upper()}
{complexity_emoji.get(analysis['complexity'], '⚪')} **Complexidade:** {analysis['complexity'].capitalize()}
{tone_emoji.get(analysis['tone'], '💭')} **Tom Esperado:** {analysis['tone'].capitalize()}
📝 **Palavras na Query:** {analysis['word_count']}
"""
    
    if analysis['keywords']:
        report += f"\n🔑 **Termos Técnicos Identificados:** {', '.join(analysis['keywords'])}"
    
    report += "\n\n**O agente adaptará a resposta para:**\n"
    
    if analysis['intent'] == 'analisar':
        report += "→ Fornecer análise crítica com riscos e recomendações\n"
    elif analysis['intent'] == 'informar':
        report += "→ Responder de forma clara e objetiva\n"
    elif analysis['intent'] == 'redigir':
        report += "→ Produzir texto formal completo e estruturado\n"
    elif analysis['intent'] == 'comparar':
        report += "→ Apresentar comparação sistemática com recomendações\n"
    elif analysis['intent'] == 'avaliar':
        report += "→ Emitir parecer técnico sobre viabilidade\n"
    
    if analysis['complexity'] == 'simples':
        report += "→ Resposta concisa e direta ao ponto\n"
    elif analysis['complexity'] == 'médio':
        report += "→ Resposta equilibrada com fundamentação moderada\n"
    else:
        report += "→ Análise aprofundada e técnica exaustiva\n"
    
    if analysis['tone'] == 'técnico':
        report += "→ Linguagem técnico-jurídica precisa\n"
    elif analysis['tone'] == 'formal':
        report += "→ Linguagem formal profissional\n"
    else:
        report += "→ Linguagem clara e acessível\n"
    
    return report
