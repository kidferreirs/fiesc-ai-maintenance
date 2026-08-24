import json

from sentence_transformers import SentenceTransformer
from llm_service import generate_response
from rag_retrieval import (
    MODEL_NAME,
    load_chunks,
    retrieve,
)

TOP_CONTEXT = 3

def build_context(results):
    # Monta o contexto com rastreabilidade
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
    # Sem evidência documental, não gera prompt
    if not results:
        return None

    context = build_context(results)

    return f"""
Você é um assistente técnico de manutenção industrial.

FAMÍLIA CANDIDATA:
{family}

PERGUNTA:
{user_question}

DOCUMENTAÇÃO RECUPERADA:
{context}

REGRAS OBRIGATÓRIAS:

1. Use somente a documentação recuperada.
2. Não use conhecimento externo.
3. Não invente causas, limites, tolerâncias ou procedimentos.
4. A família informada é uma hipótese apoiada por análise histórica,
   não um diagnóstico confirmado.
5. Se a documentação não sustentar uma conclusão, diga isso claramente.
6. Só mencione procedimentos de segurança quando eles estiverem
   explicitamente presentes na DOCUMENTAÇÃO RECUPERADA.
7. Cite sempre documento e página.
8. Retorne somente JSON válido.
9. Não use markdown.
10. Use exatamente esta estrutura:

{{
  "status": "ready",
  "family": "{family}",
  "evidence": [
    "..."
  ],
  "interpretation": "...",
  "recommended_actions": [
    "..."
  ],
  "sources": [
    {{
      "document": "...",
      "page": 0
    }}
  ]
}}
""".strip()

def prepare_rag(
    family,
    question,
    chunks,
    model,
):
    # Recupera somente documentação autorizada
    results = retrieve(
        family=family,
        chunks=chunks,
        model=model,
        user_query=question,
    )

    # Guardrail: família sem documento
    if not results:
        return {
            "status": "abstain",
            "family": family,
            "message": (
                "A família foi identificada, porém "
                "não existe documentação técnica autorizada "
                "na base para fundamentar uma recomendação."
            ),
            "sources": [],
        }

    prompt = build_grounded_prompt(
        family,
        question,
        results,
    )

    return {
        "status": "ready",
        "family": family,
        "prompt": prompt,
        "retrieved": results[:TOP_CONTEXT],
    }

def generate_rag_response(
    family,
    question,
    chunks,
    model,
):
    # Prepara o contexto antes de chamar o LLM
    prepared = prepare_rag(
        family=family,
        question=question,
        chunks=chunks,
        model=model,
    )

    # Não chama a OpenAI quando não há documento
    if prepared["status"] == "abstain":
        return prepared

    # Gera a resposta usando apenas o contexto recuperado
    llm_output = generate_response(
        prepared["prompt"]
    )

    # Converte a resposta JSON em estrutura Python
    try:
        answer = json.loads(llm_output)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "family": family,
            "message": (
                "O modelo retornou uma resposta "
                "fora do formato JSON esperado."
            ),
            "raw_answer": llm_output,
        }

    # Retorna resposta e fontes utilizadas
    return {
        "status": "ready",
        "family": family,
        "answer": answer,
        "retrieved_sources": [
            {
                "document": item["document"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
            }
            for item in prepared["retrieved"]
        ],
    }

def print_result(result):
    print("\n" + "=" * 90)
    print("FIESC - RESPOSTA RAG + LLM")
    print("=" * 90)

    print(f"\nStatus : {result['status']}")
    print(f"Família: {result['family']}")

    if result["status"] == "abstain":
        print("\nResposta:")
        print(result["message"])
        return

    if result["status"] == "error":
        print("\nErro:")
        print(result["message"])

        if result.get("raw_answer"):
            print("\nResposta bruta:")
            print(result["raw_answer"])

        return

    print("\nResposta do modelo:")
    print(
        json.dumps(
            result["answer"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nFontes recuperadas pelo sistema:")

    for source in result["retrieved_sources"]:
        print(
            f"- {source['document']} "
            f"página {source['page']} "
            f"({source['chunk_id']})"
        )

def main():
    chunks = load_chunks()

    # Carrega embeddings uma única vez
    model = SentenceTransformer(
        MODEL_NAME
    )

    tests = [
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
        result = generate_rag_response(
            family=family,
            question=question,
            chunks=chunks,
            model=model,
        )

        print_result(result)

if __name__ == "__main__":
    main()