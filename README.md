# FIESC — Assistente de Manutenção Prescritiva com IA

Prova de conceito para apoio à análise de eventos de manutenção industrial utilizando dados históricos, similaridade entre eventos, documentação técnica, RAG (Retrieval-Augmented Generation) e modelo de linguagem.

O objetivo da solução não é substituir o especialista de manutenção nem confirmar diagnósticos automaticamente. O sistema identifica uma família de falha candidata a partir do histórico e, somente quando existem evidências suficientes e documentação técnica autorizada, gera uma recomendação fundamentada.

---

## 1. Visão Geral

O projeto utiliza duas fontes principais de conhecimento:

1. **Histórico de eventos industriais**
   - 166.796 registros processados;
   - variáveis de vibração, aceleração, temperatura, frequência e RPM;
   - famílias de falhas normalizadas.

2. **Documentação técnica**
   - documentos relacionados às famílias de falhas;
   - ingestão e divisão em chunks;
   - recuperação lexical e semântica;
   - utilização das fontes recuperadas para fundamentar a resposta da IA.

O fluxo principal é:

```text
Evento industrial
      |
      v
Validação da entrada
      |
      v
Normalização / features
      |
      v
Busca de eventos similares
      |
      v
Família candidata
      |
      v
Confidence Gate
   |       |
   |       +---- confiança insuficiente ---> ABSTAIN
   |
   v
Documentação autorizada?
   |       |
   |       +---- não ---> ABSTAIN
   |
   v
Retrieval documental
      |
      v
RAG + LLM
      |
      v
Recomendação fundamentada
      |
      v
Registro da análise
```

---

## 2. Arquitetura

A solução foi organizada em camadas independentes.

### Diagrama da arquitetura

![Arquitetura da solução](docs/images/arquitetura-fiesc-ai-maintenance.png)

### Dados históricos

O dataset é carregado e normalizado para reduzir inconsistências nos rótulos de falhas.

Foram analisadas:

- distribuição temporal;
- famílias de falhas;
- variáveis disponíveis;
- similaridade entre eventos;
- comportamento dos modelos de baseline.

### Similaridade

A identificação da família candidata utiliza características numéricas do evento e comparação com eventos históricos semelhantes.

Principais componentes:

- `RobustScaler`
- `NearestNeighbors`
- filtro por RPM
- distância entre eventos
- votação das famílias dos vizinhos

O evento analisado é removido do conjunto de candidatos quando já pertence ao histórico, evitando que seja seu próprio vizinho mais próximo.

A família retornada representa uma **hipótese baseada em similaridade histórica**, e não um diagnóstico confirmado.

### Confidence Gate

A análise experimental mostrou que apenas a concordância entre vizinhos não era suficiente para permitir recomendações automáticas.

Foram avaliados 4.000 eventos.

Resultado geral da similaridade:

```text
Acerto da família: 73,50%
```

Foram então avaliadas combinações entre suporte histórico e distância média.

Um dos resultados observados foi:

```text
Suporte mínimo       : 1.00
Distância média máx. : 1.00
Cobertura            : 27,93%
Precisão             : 98,12%
```

Como a aplicação envolve manutenção industrial, foi priorizada precisão em relação à cobertura.

Quando o evento não atende ao nível mínimo definido, o pipeline retorna:

```text
status = abstain
abstain_reason = low_confidence
```

Nesse cenário nenhuma recomendação automática é produzida.

---

## 3. Base Documental

A documentação técnica é associada às famílias autorizadas através de um registro explícito.

Famílias documentadas utilizadas na POC incluem:

- rolamento — pista interna;
- rolamento — pista externa;
- elementos rolantes;
- falhas combinadas de rolamento;
- desalinhamento;
- desbalanceamento;
- correia;
- polia;
- rotor inclinado (`cocked_rotor`).

Documentos não disponíveis ou famílias sem documentação autorizada não são complementados com conhecimento externo.

Nesses casos o sistema se abstém de gerar uma recomendação e orienta o usuário a registrar um novo documento técnico para o defeito antes de solicitar uma nova recomendação automática.

---

## 4. RAG — Retrieval-Augmented Generation

A documentação é processada em chunks e recuperada de acordo com a família candidata e a pergunta técnica.

O projeto possui experimentos com:

- recuperação lexical;
- recuperação semântica;
- embeddings multilíngues;
- recuperação híbrida;
- Reciprocal Rank Fusion (RRF).

Modelo utilizado para embeddings:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

O contexto recuperado contém:

- documento;
- página;
- chunk;
- conteúdo técnico.

Somente documentação autorizada para a família candidata pode ser utilizada.

---

## 5. LLM

A camada generativa utiliza a API da OpenAI.

A LLM recebe:

- família candidata;
- pergunta;
- documentação recuperada;
- regras explícitas de segurança.

A resposta é estruturada em:

```text
Evidências encontradas
Interpretação
Ações sugeridas
Fontes
```

O prompt determina que o modelo:

- utilize exclusivamente a documentação recuperada;
- não invente procedimentos ou limites;
- diferencie hipótese de diagnóstico confirmado;
- informe quando não existem evidências suficientes;
- cite documento e página;
- priorize segurança.

A chave da API não é armazenada no repositório.

Ela deve ser configurada através de variável de ambiente.

---

## 6. Controle de Alucinação

O controle de alucinação foi tratado em várias camadas.

### Entrada inválida

Eventos sem campos obrigatórios são interrompidos antes da inferência.

Exemplo:

```text
status = invalid_input
```

### Baixa confiança histórica

A família candidata pode ser identificada, mas isso não significa que uma recomendação será produzida.

```text
status = abstain
abstain_reason = low_confidence
```

### Ausência de documentação

Mesmo que exista uma família candidata, o sistema não utiliza conhecimento externo quando não existe documentação autorizada.

Nesse caso, nenhuma chamada generativa é realizada e o sistema orienta o registro de um novo documento técnico para a família antes de uma nova tentativa de recomendação.

### Grounding

Quando a recomendação é permitida, a LLM recebe somente os chunks técnicos recuperados.

### Fontes

Documento e página utilizados são apresentados junto da recomendação.

---

## 7. Estados do Pipeline

O pipeline possui três comportamentos principais.

### READY

```text
Entrada válida
+ confiança histórica suficiente
+ documentação autorizada
= recomendação fundamentada
```

### ABSTAIN

```text
Confiança insuficiente
ou
ausência de documentação autorizada
= nenhuma recomendação automática
```

### INVALID_INPUT

```text
Entrada incompleta ou inválida
= processamento interrompido
```

---

## 8. Persistência

As análises executadas pela aplicação são registradas localmente em SQLite.

O banco permite manter rastreabilidade das execuções durante a demonstração.

São armazenadas informações como:

- status;
- evento;
- RPM;
- família candidata;
- suporte histórico;
- resultado da análise.

O arquivo local do banco não é versionado no Git.

---

## 9. Interface

Foi criada uma interface utilizando Streamlit para permitir interação mínima com a solução.

A interface possui:

- nova análise;
- histórico das análises;
- resumo histórico da família candidata;
- quantidade de registros da família;
- quantidade de registros no mesmo RPM;
- período histórico e regime operacional analisado;
- visualização dos eventos similares;
- evidências técnicas;
- interpretação;
- ações sugeridas;
- fontes utilizadas.

Executar:

```bash
streamlit run src/app.py
```

---

## 10. Arquitetura Proposta para Ambiente Industrial

A POC foi desenvolvida de forma modular para permitir evolução para um ambiente industrial sem acoplar a interface, o mecanismo de similaridade, a recuperação documental e a camada generativa.

Uma possível arquitetura de implantação é:

```text
Sensores / PLC / Sistemas industriais
              |
              v
     Camada de ingestão de dados
              |
              v
 Banco de dados corporativo / histórico
              |
              v
      Serviço de análise Python
       |                  |
       |                  +---- Similaridade histórica
       |                  +---- Confidence Gate
       |                  +---- Base documental / RAG
       |                  +---- LLM
       |
       v
          API interna
       |             |
       v             v
Dashboard       Sistema de manutenção
       \             /
        \           /
         v         v
        Banco de auditoria
```

### Princípios para implantação

- execução do serviço de análise em ambiente controlado;
- integração com fontes industriais através de APIs ou camada de ingestão;
- acesso controlado ao histórico operacional;
- armazenamento seguro de credenciais em serviço de secrets;
- autenticação e controle de acesso;
- versionamento da documentação técnica;
- logs e trilha de auditoria das análises;
- observabilidade do serviço e dos modelos;
- aprovação humana antes de intervenções críticas;
- possibilidade de execução containerizada;
- separação entre inferência, persistência e interface.

A recomendação da IA permanece como apoio à decisão. O desenho não prevê atuação autônoma da LLM sobre máquinas, PLCs ou sistemas de controle.

---

## 11. Instalação

Clone o repositório:

```bash
git clone https://github.com/kidferreirs/fiesc-ai-maintenance.git
cd fiesc-ai-maintenance
```

Crie o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Configure a chave da OpenAI:

```bash
export OPENAI_API_KEY="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"
```

Execute a aplicação:

```bash
streamlit run src/app.py
```

---

## 12. Principais Arquivos

```text
src/
├── app.py
├── pipeline.py
├── database.py
├── llm_service.py
├── rag_response.py
├── rag_retrieval.py
├── semantic_retrieval.py
├── hybrid_retrieval.py
├── document_ingestion.py
├── document_chunking.py
├── document_registry.py
├── confidence_analysis.py
├── group_validation.py
├── ml_baseline.py
└── similarity_experiment.py
```

### `pipeline.py`

Orquestra o fluxo principal da análise.

### `app.py`

Interface Streamlit.

### `database.py`

Persistência das análises.

### `rag_retrieval.py`

Recuperação documental utilizada pelo RAG.

### `rag_response.py`

Construção do contexto e resposta fundamentada.

### `llm_service.py`

Integração com o modelo de linguagem.

### `confidence_analysis.py`

Experimento utilizado para avaliar os critérios do confidence gate.

---

## 13. Resultados e Decisões Técnicas

Durante o desenvolvimento foram testadas diferentes estratégias.

A validação por grupos apresentou desempenho significativamente inferior quando comparada a divisões mais próximas dos dados históricos:

```text
Accuracy média : 17,78%
Macro F1       : 16,47%
Weighted F1    : 20,14%
```

Esse resultado indica dificuldade de generalização para grupos independentes e foi considerado uma limitação importante da abordagem puramente supervisionada.

A análise de similaridade apresentou:

```text
Acerto geral da família: 73,50%
```

Entretanto, a precisão aumenta significativamente quando o sistema aceita somente casos de maior confiança.

Com:

```text
support = 1.00
mean_distance <= 1.00
```

o experimento apresentou:

```text
Precisão  : 98,12%
Cobertura : 27,93%
```

Por esse motivo a POC adota uma estratégia conservadora: **é preferível não recomendar do que produzir uma recomendação baseada em evidência histórica fraca.**

---

## 14. Testes Funcionais

Foram validados três cenários principais.

### Caso positivo

Evento com forte evidência histórica e documentação disponível:

```text
Família candidata  : correia
Suporte histórico  : 100%
Status              : ready
```

O sistema recuperou documentação técnica e produziu recomendação fundamentada.

### Baixa confiança

Evento cuja similaridade histórica não atingiu o critério definido:

```text
Status         : abstain
Abstain reason : low_confidence
```

Nenhuma recomendação automática foi gerada.

### Entrada incompleta

Evento sem campo obrigatório:

```text
Status : invalid_input
```

O pipeline interrompeu o processamento antes da inferência.

---

## 15. Limitações

Esta solução é uma prova de conceito.

Entre as limitações identificadas:

- capacidade limitada de generalização para grupos completamente independentes;
- cobertura reduzida quando são utilizados critérios de confiança mais rigorosos;
- documentação disponível apenas para parte das famílias;
- ausência de validação em ambiente industrial real;
- ausência de integração direta com sensores ou sistemas industriais;
- thresholds do confidence gate derivados experimentalmente deste conjunto de dados.

A solução não deve ser interpretada como sistema autônomo de diagnóstico ou controle industrial.

---

## 16. Possíveis Evoluções

Como evolução da arquitetura:

- API para integração com sistemas externos;
- banco de dados corporativo;
- autenticação e controle de acesso;
- vector database para documentação;
- monitoramento contínuo;
- integração com sensores;
- avaliação humana das recomendações;
- feedback loop;
- observabilidade de modelos;
- deploy containerizado;
- versionamento dos documentos e modelos.

---

## 17. Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- OpenAI API
- Streamlit
- SQLite
- Git / GitHub

---

## Conclusão

A POC combina histórico industrial e documentação técnica para apoiar a investigação de eventos de manutenção.

A arquitetura foi desenhada para separar três conceitos:

1. **similaridade histórica gera uma hipótese;**
2. **documentação técnica fornece fundamentação;**
3. **a LLM organiza e apresenta a recomendação.**

Quando as evidências não são suficientes, o comportamento esperado do sistema é se abster e encaminhar o caso para avaliação técnica.

Quando a família candidata não possui documentação autorizada, o sistema também se abstém e orienta o registro de um novo documento técnico antes de permitir uma recomendação automática.

Essa decisão reduz o risco de transformar similaridade estatística ou geração de linguagem em diagnóstico industrial não fundamentado.
