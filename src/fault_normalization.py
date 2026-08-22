import re

DOCUMENTED_FAMILIES = {
    "rolamento_inner",
    "rolamento_outer",
    "rolamento_ball",
    "rolamento_combination",
    "desalinhado",
    "desbalanceado",
    "correia",
    "polia",
    "cocked_rotor",
}

STATE_FAMILIES = {
    "normal",
    "baseline",
    "teste",
    "acelerando",
    "motor_desligado",
}

def normalize_text(value):
    """Padroniza texto antes da identificação da família."""

    value = str(value).strip().lower()

    # Corrige erros de digitação observados no dataset
    replacements = {
        "mortor_desligado": "motor_desligado",
        "normla": "normal",
        "desabalanceado": "desbalanceado",
        "desbanlanceado": "desbalanceado",
        "ddesbalanceado": "desbalanceado",
        "dedesbalanceado": "desbalanceado",
        "desabanceado": "desbalanceado",
        "cockecocked": "cocked",
    }

    for wrong, correct in replacements.items():
        value = value.replace(wrong, correct)
    return value

def identify_family(value):
    """Identifica a família física sem alterar o rótulo original."""

    value = normalize_text(value)

    # Estados operacionais
    if "motor_desligado" in value:
        return "motor_desligado"

    if "baseline" in value:
        return "baseline"

    if "acelerando" in value:
        return "acelerando"

    if "teste" in value or value in {"new_tes"}:
        return "teste"

    if "normal" in value:
        return "normal"

    # Falhas de rolamento
    if "rolamento_inner" in value:
        return "rolamento_inner"

    if "rolamento_outer" in value:
        return "rolamento_outer"

    if "rolamento_ball" in value:
        return "rolamento_ball"

    if (
        "rolamento_combination" in value
        or "rolamento_comb" in value
    ):
        return "rolamento_combination"

    # Falhas mecânicas
    if "desalinhado" in value:
        return "desalinhado"

    if (
        "desbalanceado" in value
        or "desbalanceamento" in value
    ):
        return "desbalanceado"

    if (
        "eccentric_rotor" in value
        or "eccentric" in value
    ):
        return "eccentric_rotor"

    if (
        "cocked_rotor" in value
        or "cocked" in value
    ):
        return "cocked_rotor"

    if "correia" in value:
        return "correia"

    if "polia" in value:
        return "polia"

    if "ventoinha" in value:
        return "ventoinha"

    if "falta_fase" in value:
        return "falta_fase"

    return "unknown"

def normalize_fault(value):
    """Retorna informações normalizadas sobre o rótulo histórico."""

    original = str(value).strip()
    family = identify_family(original)

    if family in STATE_FAMILIES:
        condition_type = "state"
    elif family == "unknown":
        condition_type = "unknown"
    else:
        condition_type = "problem"

    return {
        "original": original,
        "family": family,
        "condition_type": condition_type,
        "documented": family in DOCUMENTED_FAMILIES,
    }

def main():
    # Pequenos testes manuais para validar a normalização
    examples = [
        "cocked_rotor_2_pos_2",
        "new_rolamento_inner_3",
        "normal_carga_3_3",
        "mortor_desligado_novo",
        "desbanlanceado_carga_3_2",
        "new_falta_fase_2",
        "ventoinha_adxl_0",
        "valor_desconhecido",
    ]

    print("=" * 80)
    print("FIESC - TESTE DE NORMALIZAÇÃO DE FAULT")
    print("=" * 80)

    for example in examples:
        result = normalize_fault(example)

        print(f"\nOriginal       : {result['original']}")
        print(f"Família        : {result['family']}")
        print(f"Tipo           : {result['condition_type']}")
        print(f"Documentado    : {result['documented']}")

    print("\n" + "=" * 80)
    print("TESTE CONCLUÍDO")
    print("=" * 80)

if __name__ == "__main__":
    main()