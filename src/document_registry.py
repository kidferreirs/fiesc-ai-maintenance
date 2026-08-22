from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

# Associação explícita entre família de falha e documento técnico
DOCUMENT_REGISTRY = {
    # Rolamentos - Doc1 cobre os principais modos de falha
    "rolamento_inner": DOCS_DIR / "Doc1.pdf",
    "rolamento_outer": DOCS_DIR / "Doc1.pdf",
    "rolamento_ball": DOCS_DIR / "Doc1.pdf",
    "rolamento_combination": DOCS_DIR / "Doc1.pdf",

    # Demais famílias documentadas
    "desalinhado": DOCS_DIR / "Doc2.pdf",
    "desbalanceado": DOCS_DIR / "Doc3.pdf",
    "correia": DOCS_DIR / "Doc4.pdf",
    "polia": DOCS_DIR / "Doc5.pdf",
    "cocked_rotor": DOCS_DIR / "Doc6.pdf",
}

def get_document(fault_family):
    # Retorna o documento da família, quando existir
    return DOCUMENT_REGISTRY.get(fault_family)

def has_document(fault_family):
    # Verifica se existe documentação cadastrada
    document = get_document(fault_family)

    return document is not None and document.exists()

def main():
    print("=" * 80)
    print("FIESC - REGISTRO DE DOCUMENTOS")
    print("=" * 80)

    for family, document in DOCUMENT_REGISTRY.items():
        status = "OK" if document.exists() else "NÃO ENCONTRADO"

        print(f"{family:<25} -> {document.name:<10} [{status}]")

    print("\n" + "=" * 80)
    print("VALIDAÇÃO CONCLUÍDA")
    print("=" * 80)

if __name__ == "__main__":
    main()