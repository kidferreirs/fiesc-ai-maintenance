from sentence_transformers import SentenceTransformer

from rag_retrieval import (
    MODEL_NAME,
    load_chunks,
    retrieve,
)

TOP_CONTEXT = 3

def build_context(results):
    # Monta contexto mantendo fonte e página
    context_parts = []

    for result in results[:TOP_CONTEXT]:
        source = (
            f"[Fonte: {result['document']} "
            f"- página {result['page']}]"
        )

        context_parts.append(
            f"{source}\n{result['text']}"
        )

    return "\n\n".join(context_parts)

def build_grounded_prompt(
    family,
    user_question,
    results,
):
    # Sem documento, o sistema deve se abster
    if not results:
        return None

    context = build_context(results)

    prompt = f"""
Você é um assistente técnico de manutenção industrial.

Sua tarefa é apoiar a análise de uma ocorrência identificada
como pertencente à família:

{family}

PERGUNTA:
{user_question}

DOCUMENTAÇÃO RECUPERADA:
{context}

REGRAS OBRIGATÓRIAS:

1. Use exclusivamente as informações presentes na
   DOCUMENTAÇÃO RECUPERADA.

2. Não invente causas, procedimentos, limites,
   tolerâncias ou valores técnicos.

3. Diferencie hipótese de diagnóstico confirmado.

4. Não afirme que a falha está confirmada apenas
   porque existe similaridade histórica.

5. Quando a documentação não sustentar alguma
   conclusão, informe explicitamente:
   "A documentação recuperada não fornece evidência
   suficiente para essa conclusão."

6. Priorize segurança antes de recomendar intervenção.

7. Sempre informe documento e página utilizados.

8. Estruture a resposta em:
   - Evidências encontradas
   - Interpretação
   - Ações sugeridas
   - Fontes

9. Seja objetivo e técnico.

Não utilize conhecimento externo à documentação.
""".strip()

    return prompt

def prepare_rag(
    family,
    question,
    chunks,
    model,
):
    # Recupera evidência documental
    results = retrieve(
        family=family,
        chunks=chunks,
        model=model,
        user_query=question,
    )

    # Guardrail de ausência documental
    if not results:
        return {
            "status": "abstain",
            "family": family,
            "message": (
                "A família foi identificada, porém "
                "não existe documentação técnica "
                "autorizada na base para fundamentar "
                "uma recomendação."
            ),
            "sources": [],
            "prompt": None,
        }

    prompt = build_grounded_prompt(
        family,
        question,
        results,
    )

    sources = [
        {
            "document": item["document"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "rrf_score": item["rrf_score"],
        }
        for item in results[:TOP_CONTEXT]
    ]

    return {
        "status": "ready",
        "family": family,
        "sources": sources,
        "prompt": prompt,
    }

def print_rag_result(result):
    print("\n" + "=" * 90)
    print("FIESC - CONTEXTO FUNDAMENTADO PARA LLM")
    print("=" * 90)

    print(f"\nStatus : {result['status']}")
    print(f"Família: {result['family']}")

    if result["status"] == "abstain":
        print("\nResposta:")
        print(result["message"])
        return

    print("\nFontes selecionadas:")

    for source in result["sources"]:
        print(
            f"- {source['document']} "
            f"página {source['page']} "
            f"({source['chunk_id']})"
        )

    print("\nPROMPT GERADO")
    print("-" * 90)
    print(result["prompt"])

def main():
    chunks = load_chunks()

    # Carrega o modelo somente uma vez
    model = SentenceTransformer(
        MODEL_NAME
    )

    tests = [
        (
            "rolamento_inner",
            (
                "O equipamento apresenta vibração crescente. "
                "O que devo verificar?"
            ),
        ),
        (
            "correia",
            (
                "A correia está escorregando e oscilando "
                "durante a operação. Como devo proceder?"
            ),
        ),
        (
            "falta_fase",
            "Como devo corrigir esta condição?",
        ),
    ]

    for family, question in tests:
        result = prepare_rag(
            family=family,
            question=question,
            chunks=chunks,
            model=model,
        )

        print_rag_result(result)

if __name__ == "__main__":
    main()