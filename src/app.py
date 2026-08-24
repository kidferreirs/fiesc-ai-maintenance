import json
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from database import (
    init_db,
    list_analyses,
    save_analysis,
)
from pipeline import (
    analyze_event,
    load_history,
)
from rag_retrieval import (
    MODEL_NAME,
    load_chunks,
)

st.set_page_config(
    page_title="FIESC - Manutenção Prescritiva",
    page_icon="⚙️",
    layout="wide",
)

@st.cache_data
def get_history():
    # Histórico operacional
    return load_history()

@st.cache_data
def get_chunks():
    # Base documental
    return load_chunks()

@st.cache_resource
def get_embedding_model():
    # Modelo carregado apenas uma vez
    return SentenceTransformer(MODEL_NAME)

def show_similar_events(events):
    if not events:
        return

    df = pd.DataFrame(events)

    columns = [
        "id",
        "created_at",
        "fault_family",
        "rpm",
        "distance",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    st.dataframe(
        df[available],
        use_container_width=True,
        hide_index=True,
    )

def show_recommendation(recommendation):
    if not recommendation:
        return

    st.subheader("Evidências")

    for evidence in recommendation.get(
        "evidence",
        [],
    ):
        st.write(f"• {evidence}")

    st.subheader("Interpretação")

    st.write(
        recommendation.get(
            "interpretation",
            "Não disponível.",
        )
    )

    st.subheader("Ações sugeridas")

    for action in recommendation.get(
        "recommended_actions",
        [],
    ):
        st.write(f"• {action}")

    st.subheader("Fontes")

    sources = recommendation.get(
        "sources",
        [],
    )

    for source in sources:
        st.write(
            f"📄 {source['document']} "
            f"— página {source['page']}"
        )

def show_result(result):
    status = result.get("status")

    if status == "invalid_input":
        st.error(result["message"])
        return

    if status == "error":
        st.error(result.get("message", "Erro na análise."))
        return

    if status == "abstain":
        st.warning(result["message"])

    if status == "state":
        st.info(result["message"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Família candidata",
        result.get(
            "candidate_family",
            "-",
        ),
    )

    support = result.get(
        "historical_support"
    )

    if support is not None:
        col2.metric(
            "Suporte histórico",
            f"{support * 100:.0f}%",
        )
    else:
        col2.metric(
            "Suporte histórico",
            "-",
        )

    col3.metric(
        "RPM",
        result.get("rpm", "-"),
    )

    st.caption(
        "Suporte histórico representa a proporção dos "
        "vizinhos Top-5 pertencentes à família candidata. "
        "Não representa probabilidade ou diagnóstico confirmado."
    )

    st.subheader("Eventos históricos semelhantes")

    show_similar_events(
        result.get(
            "similar_events",
            [],
        )
    )

    if status == "ready":
        show_recommendation(
            result.get(
                "recommendation"
            )
        )

    elif status == "abstain":
        reason = result.get("abstain_reason")

        # Explica por que o pipeline interrompeu a análise
        if reason == "low_confidence":
            st.info(
                "Nenhuma chamada generativa foi realizada porque "
                "a evidência histórica não atingiu o nível mínimo de confiança."
            )
        else:
            st.info(
                "Nenhuma chamada generativa foi realizada porque "
                "não existe documentação técnica autorizada para fundamentar "
                "uma recomendação."
            )

def main():
    init_db()

    st.title(
        "⚙️ Manutenção Prescritiva com IA"
    )

    st.write(
        "Busca por similaridade histórica, recuperação "
        "documental e recomendação fundamentada com IA Generativa."
    )

    tab_analysis, tab_history = st.tabs(
        [
            "🔎 Nova análise",
            "📚 Histórico",
        ]
    )

    with tab_analysis:
        st.subheader("Novo evento")

        st.write(
            "Insira as métricas do evento. "
            "O campo `fault` não é utilizado na inferência."
        )

        example_event = {
            "id": 114387,
            "created_at": (
                "2026-06-01 "
                "21:32:53.911176+00:00"
            ),
            "z_rms_velocity_mm_s": 1.517,
            "temperature_c": 24.69,
            "x_rms_velocity_mm_s": 2.0,
            "z_peak_acceleration_g": 0.484,
            "x_peak_acceleration_g": 0.631,
            "z_peak_vel_comp_freq_hz": 61.0,
            "x_peak_vel_comp_freq_hz": 61.0,
            "z_rms_acceleration_g": 0.09,
            "x_rms_acceleration_g": 0.114,
            "z_kurtosis": 2.392,
            "x_kurtosis": 2.77,
            "z_crest_factor": 3.747,
            "x_crest_factor": 4.269,
            "z_high_freq_rms_accel_g": 0.129,
            "x_high_freq_rms_accel_g": 0.147,
            "rpm": 1000.0,
        }

        event_text = st.text_area(
            "JSON do evento",
            value=json.dumps(
                example_event,
                ensure_ascii=False,
                indent=2,
            ),
            height=430,
        )

        if st.button(
            "Analisar evento",
            type="primary",
            use_container_width=True,
        ):
            try:
                event = json.loads(
                    event_text
                )
            except json.JSONDecodeError:
                st.error(
                    "O conteúdo informado não é um JSON válido."
                )
                return

            with st.spinner(
                "Analisando histórico e documentação..."
            ):
                history = get_history()
                chunks = get_chunks()
                model = get_embedding_model()

                result = analyze_event(
                    event=event,
                    history=history,
                    chunks=chunks,
                    embedding_model=model,
                )

                analysis_id = save_analysis(
                    result
                )

            st.success(
                f"Análise concluída e registrada "
                f"com ID {analysis_id}."
            )

            show_result(result)

    with tab_history:
        st.subheader(
            "Últimas análises"
        )

        analyses = list_analyses(
            limit=20
        )

        if not analyses:
            st.info(
                "Nenhuma análise registrada."
            )
        else:
            history_df = pd.DataFrame(
                analyses
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

if __name__ == "__main__":
    main()