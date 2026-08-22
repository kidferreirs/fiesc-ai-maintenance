import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-5.4-mini"

def get_client():
    # A chave é lida somente do ambiente
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada no arquivo .env."
        )

    return OpenAI(api_key=api_key)

def generate_response(prompt):
    # Executa somente após os guardrails do RAG
    client = get_client()

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    text = response.output_text.strip()

    return text

def main():
    # Teste mínimo da conexão com a API
    prompt = """
Responda somente com este JSON válido:

{
  "status": "ok",
  "message": "Conexão com o modelo funcionando."
}
""".strip()

    print("=" * 80)
    print("FIESC - TESTE OPENAI")
    print("=" * 80)

    try:
        result = generate_response(prompt)

        print("\nResposta:")
        print(result)

    except Exception as error:
        print("\nErro ao chamar a OpenAI:")
        print(error)

if __name__ == "__main__":
    main()