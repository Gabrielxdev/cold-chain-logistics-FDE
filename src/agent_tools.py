# MAPA DO FLUXO ENTRE OS ARQUIVOS
# 1. ui.py recebe a pergunta e chama fde_agent.stream(...).
# 2. orchestrator.py envia o histórico ao modelo no nó reasoner.
# 3. O modelo pode pedir uma ferramenta pelo nome e fornecer seus argumentos.
# 4. O nó tools (ToolNode) encontra a ferramenta deste arquivo e a executa.
# 5. O texto retornado vira conteúdo de uma ToolMessage no estado do grafo.
# 6. O grafo volta ao reasoner: o modelo lê o resultado e decide o próximo passo.
# 7. ui.py acompanha os eventos, registra a auditoria e exibe a resposta.
#
# PAPEL DESTE ARQUIVO: acessar dados, sem escolher a sequência do atendimento.
# query_telemetry_db        -> SQL Server: dados da frota.
# fetch_corridor_conditions -> API HTTP: clima e classificação calculada do corredor.
# search_compliance_sop    -> Pinecone: trechos dos procedimentos internos (SOPs).
# As funções retornam texto ao grafo; não escrevem diretamente no chat nem nos logs.
#
# ORDEM PARA ESTUDAR: leia este mapa, depois reasoning_node e a montagem do grafo
# em orchestrator.py; volte às ferramentas para entender como cada consulta funciona.

# Biblioteca padrão para acessar recursos do sistema operacional, como variáveis de ambiente.
import os
# Pacote padrão para trabalhar com URLs, por exemplo, analisar ou abrir endereços web.
import urllib
# Biblioteca externa para fazer requisições HTTP a APIs e serviços web.
import requests
# Classe da biblioteca padrão que facilita montar e manipular caminhos de arquivos e pastas.
from pathlib import Path
# Função que carrega variáveis definidas em um arquivo .env para o ambiente do processo.
from dotenv import load_dotenv
# create_engine configura a conexão com um banco de dados; text permite escrever SQL textual.
from sqlalchemy import create_engine, text

# Decorador do LangChain que transforma uma função Python em uma ferramenta utilizável por um agente.
from langchain_core.tools import tool
# Integração do LangChain que converte textos em vetores (embeddings) usando um modelo da OpenAI.
from langchain_openai import OpenAIEmbeddings
# Integração que armazena e pesquisa vetores em um índice do Pinecone.
from langchain_pinecone import PineconeVectorStore

# ==========================================
# 1. INICIALIZAÇÃO DOS RECURSOS COMPARTILHADOS
# ==========================================
# Este código fora das funções roda ao carregar o módulo, antes de atender perguntas.
# Na interface, a cadeia de imports é ui.py -> orchestrator.py -> agent_tools.py.
# Imports comuns reutilizam o módulo já carregado no processo; não executam todas
# estas instruções novamente a cada chamada de ferramenta.
# Definir uma função prepara seu uso; o corpo dela só roda quando for chamada.
#
# Localiza a raiz do projeto para encontrar o arquivo .env.
# __file__ é o caminho deste arquivo; resolve() o torna absoluto e parent obtém a pasta src.
script_dir = Path(__file__).resolve().parent
# Como agent_tools.py está dentro de src, a pasta pai de src é a raiz do projeto.
project_root = script_dir.parent

# Carrega as configurações do arquivo .env localizado na raiz do projeto.
load_dotenv(dotenv_path=project_root / ".env")

# Lê a chave do Pinecone das variáveis de ambiente; será None se não estiver configurada.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Interrompe a inicialização logo no começo se a chave obrigatória do Pinecone estiver ausente.
# Isso evita descobrir só depois, durante uma operação, que a conexão não pode ser autenticada.
# Como a checagem ocorre no import, sua falha também impede carregar o orquestrador
# e a interface, mesmo que a pergunta precisasse apenas de SQL ou clima.
if not PINECONE_API_KEY:
    raise ValueError(
        "CRITICAL: Ensure PINECONE_API_KEY is present in your active environment."
    )

# Lê o modelo escolhido; se a variável não existir, usa LOCAL.
# strip() remove espaços nas pontas e upper() padroniza o texto em letras maiúsculas.
EMBEDDINGS_MODEL_SETTING = os.getenv("Embeddings_model", "LOCAL").strip().upper()


# FUNÇÃO AUXILIAR: não tem @tool e não é oferecida ao modelo de linguagem.
# Entrada: identificador do modelo local. Saída: objeto que transforma texto em vetores.
# É usada na preparação do retriever, não como uma etapa escolhida pelo agente.
def get_cached_huggingface_embeddings(model_name: str):
    """Carrega o modelo Hugging Face uma vez e reutiliza a instância em reruns."""
    # Importar Streamlit aqui evita exigir seu uso quando esta função não for chamada.
    import streamlit as st

    # cache_resource guarda objetos pesados (como modelos) na memória do processo.
    # O nome do modelo é um argumento da função interna e também faz parte da chave
    # do cache: modelos diferentes recebem instâncias diferentes.
    @st.cache_resource(show_spinner=False)
    def _load_model(name: str):
        # Função interna: separa a criação custosa do modelo da obtenção via cache.
        # O cache pode ser compartilhado entre sessões; não é o histórico da conversa
        # e não é permanente em disco: pode ser limpo ou perdido ao encerrar o processo.
        # Este código só roda quando ainda não existe um modelo em cache para "name".
        print(f"🧠 MEMORY SEED: Permanently caching local model [{name}] in global RAM...")
        from langchain_huggingface import HuggingFaceEmbeddings

        # Configura o modelo para processar os textos na CPU.
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": "cpu"},
        )

    # Retorna a instância já carregada, ou carrega e armazena a primeira vez.
    return _load_model(model_name)


# 2. Lê as configurações de conexão com o SQL Server do ambiente (incluindo o .env).
# Se SQL_SERVER_HOST não estiver definido, tenta conectar ao banco na própria máquina.
db_host = os.getenv("SQL_SERVER_HOST", "localhost")
# Portas são texto nas variáveis de ambiente; o padrão do SQL Server é 1433.
db_port = os.getenv("SQL_SERVER_PORT", "1433")
# Usa o nome de uma conta de leitura como padrão; as permissões efetivas são
# definidas no SQL Server, e não pelo nome da conta ou por este comentário.
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
# A senha não tem valor padrão: deve ser configurada com segurança no ambiente ou no .env.
db_password = os.getenv("SQL_AGENT_PASSWORD")

# INDEX_NAME identifica o índice do Pinecone onde os vetores serão guardados e pesquisados.
# Este arquivo consulta conteúdo já indexado; não faz a ingestão dos documentos.
# A configuração escolhe índices separados para OpenAI e para o modelo local.
# O modelo e a dimensão devem ser compatíveis com os usados na indexação.
# O modelo de embeddings representa textos como números para buscar semelhança;
# ele é diferente do LLM configurado no orquestrador, que decide e redige respostas.
if EMBEDDINGS_MODEL_SETTING == "OPENAI":
    print("🤖 Mode: Connecting to Cloud OpenAI Index (1536 Dim Space)...")
    embeddings = OpenAIEmbeddings()
    INDEX_NAME = "fde-sop-index-openai"
else :
    # Lê o identificador do modelo Hugging Face; usa BAAI/bge-m3 se não houver configuração.
    local_model_target = os.getenv("Local_Embedding_Model", "BAAI/bge-m3").strip()
    
    # "Local" se refere ao cálculo dos embeddings na máquina; o índice continua
    # no Pinecone. A dimensão impressa é fixa, não uma validação do modelo escolhido.
    print(f"🤗 Mode: Connecting to Local Fallback [{local_model_target}] Index (1024 Dim Space)...")   

    try:
        # Streamlit pode estar instalado sem que o aplicativo esteja rodando no seu runtime.
        import streamlit as st

        # Verifica se este processo está sendo executado dentro do runtime do Streamlit.
        # Quando a interface é atualizada, o Streamlit pode executar este código novamente;
        # nesse caso, a função abaixo reutiliza os embeddings guardados em cache, evitando
        # carregar o modelo Hugging Face de novo. Fora do runtime, o bloco seguinte carrega
        # o modelo diretamente, pois o cache do Streamlit não está disponível.
        if st.runtime.exists():
            embeddings = get_cached_huggingface_embeddings(local_model_target)
        else:
            # Em execução por terminal ou script comum, carrega o modelo sem depender do cache do Streamlit.
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=local_model_target,
                model_kwargs={"device": "cpu"},
            )
    except ImportError:
        # Se Streamlit não estiver instalado, ainda permite usar os embeddings locais diretamente.
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=local_model_target,
            model_kwargs={"device": "cpu"},
        )

    INDEX_NAME = "fde-sop-index-local"

# Conecta o armazenamento vetorial ao índice escolhido e informa qual modelo gera os vetores.
# Quando search_compliance_sop chamar o retriever, a pergunta será vetorizada com
# este objeto embeddings e comparada com os vetores que já existem no índice.
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
# Cria um recuperador que consulta o índice para encontrar conteúdo relevante a uma pergunta.
# k=2 limita cada busca aos dois trechos mais semelhantes, que podem ser usados como contexto.
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# ==========================================
# 2. FERRAMENTAS EXECUTADAS PELO NÓ tools DO ORQUESTRADOR
# ==========================================

# @tool transforma a função em um objeto de ferramenta. O nome, os argumentos
# tipados e a docstring (texto entre aspas triplas) descrevem seu uso para o modelo.
# O orquestrador oferece essa descrição por bind_tools e registra a execução
# por ToolNode. Apenas adicionar @tool não a inclui na lista fde_tools.
# As docstrings abaixo participam dessa descrição e foram preservadas.
#
# FERRAMENTA 1 — entrada: sql_query (texto SQL); saída: colunas e linhas em texto.
# Exemplo ilustrativo: name="query_telemetry_db", id="call_1" e
# args={"sql_query": "SELECT TOP 2 Latitude, Longitude FROM FDE_VIEWS.VW_ACTIVE_FLEET"}.
# ToolNode usa args para preencher sql_query; o ID associa a saída à solicitação.
@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the FDE_VIEWS.VW_ACTIVE_FLEET view.
    Columns available:
    Timestamp, Latitude, Longitude, Current_Temperature_C, Cargo_Condition_Code,
    Risk_Classification, Delay_Probability, Port_Congestion_Level, Route_Risk_Index.
    Always write standard T-SQL queries.
    """
    # Monta os dados que o driver ODBC precisa para abrir uma conexão com o SQL Server.
    connection_string = (
        # Informa qual driver instalado no sistema fará a comunicação com o SQL Server.
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        # Define o endereço do servidor e a porta onde o banco recebe conexões.
        f"SERVER={db_host},{db_port};"
        # Seleciona o banco inicial da conexão.
        f"DATABASE=master;"
        # Fornece o nome do usuário do banco.
        f"UID={db_user};"
        # Fornece a senha correspondente ao usuário.
        f"PWD={db_password};"
        # Desativa a criptografia da conexão.
        f"Encrypt=no;"
        # Aceita o certificado do servidor sem validar sua cadeia de confiança.
        f"TrustServerCertificate=yes;"
    )

    # Escapa caracteres da string ODBC para incluí-la na URL; não criptografa a senha.
    params = urllib.parse.quote_plus(connection_string)
    
    # Cria um mecanismo SQLAlchemy; a conexão efetiva é aberta quando a consulta é executada.
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    
    try:
        # Remove espaços e padroniza maiúsculas para verificar o início da consulta.
        # Esta é uma checagem simples de prefixo, não uma análise completa da instrução SQL.
        # A view citada na docstring orienta o modelo, mas este if não verifica
        # quais tabelas/views foram usadas. O banco aplica as permissões do usuário.
        if not sql_query.strip().upper().startswith("SELECT"):
            return "SECURITY BLOCK: Only SELECT operations are authorized on this view."
            
        # Abre a conexão e garante que ela seja fechada ao sair deste bloco.
        with engine.connect() as conn:
            # Converte o texto em uma expressão SQLAlchemy e executa a consulta recebida.
            cursor = conn.execute(text(sql_query))
            # Lê os nomes das colunas retornadas para incluí-los na resposta.
            columns = list(cursor.keys())
            # Lê no máximo 10 registros para limitar o texto devolvido ao modelo.
            # fetchmany não acrescenta TOP à consulta nem limita o trabalho do SQL Server.
            rows = cursor.fetchmany(10)
            
            # Se a consulta não encontrou registros, devolve uma mensagem clara.
            if not rows:
                return "No records matched the query criteria."
                
            # Começa a resposta listando os nomes das colunas.
            formatted_output = f"COLUMNS: {', '.join(columns)}\n"
            # Converte cada registro em texto e o acrescenta em uma linha da resposta.
            for row in rows:
                formatted_output += str(tuple(row)) + "\n"
                
            # O return encerra esta chamada. ToolNode coloca este texto em uma
            # ToolMessage, e o grafo volta ao reasoner para interpretar os dados.
            # ui.py também pode exibir esse mesmo texto como saída bruta da ferramenta.
            return formatted_output
    # Exceções dentro do try viram texto para o modelo avaliar, em vez de uma
    # resposta final automática ao usuário. A criação do engine acima fica fora do try.
    except Exception as e:
        return f"Database Error: {str(e)}"

# FERRAMENTA 2 — entrada: latitude e longitude numéricas; saída: relatório textual.
# O modelo pode usar coordenadas obtidas na ferramenta SQL para montar args:
# {"latitude": 33.77, "longitude": -118.19}. Essa passagem não é automática:
# o resultado SQL volta ao reasoner, e o modelo precisa solicitar esta nova chamada.
@tool
def fetch_corridor_conditions(latitude: float, longitude: float) -> str:
    """
    Fetches real-time weather and corridor conditions from a live REST API for given GPS coordinates.
    Provides temperature, wind speed, and computed corridor congestion index.
    """
    try:
        # Monta a URL da API com as coordenadas informadas para pedir a previsão atual.
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        # timeout=6 configura a espera de conexão/leitura da requisição HTTP;
        # não é um limite total de seis segundos para toda a execução do agente.
        response = requests.get(url, timeout=6)
        # Gera um erro se a API responder com um código HTTP de falha.
        response.raise_for_status()
        
        # Converte o corpo JSON em um dicionário e extrai o bloco de clima atual.
        payload = response.json().get("current_weather", {})
        # Lê a temperatura; usa "N/A" se o campo não vier na resposta.
        temp = payload.get("temperature", "N/A")
        # Lê a velocidade do vento; usa 0.0 se o campo não vier.
        wind = payload.get("windspeed", 0.0)
        
        # Esta é uma regra local de exemplo: o vento determina o índice de risco.
        # O índice 8.5/2.5 não é uma medição de trânsito fornecida pela API de clima.
        congestion_index = 8.5 if wind > 10.0 else 2.5
        # Gera uma descrição textual correspondente ao mesmo limite de vento.
        status_note = "High Transit Disruption" if wind > 10.0 else "Corridor Normal"
        
        # Monta um texto legível com coordenadas, clima e classificação do corredor.
        return (
            f"--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temp: {temp}°C | Wind Speed: {wind} km/h\n"
            f"Corridor Risk: {status_note} (Congestion Index: {congestion_index}/10)\n"
            f"-------------------------------"
        )
    # Se a chamada, resposta ou leitura dos dados falhar, informa o motivo.
    except Exception as e:
        return f"Corridor API Communication Failure: {str(e)}"

# FERRAMENTA 3 — entrada: query (pergunta de busca); saída: trechos dos documentos.
# Exemplo de args: {"query": "What are the temperature rules for fresh perishables?"}.
# Esta é a recuperação de contexto do RAG: buscar material para embasar a resposta.
# A síntese acontece depois, quando o orquestrador devolve os trechos ao LLM.
@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches enterprise Standard Operating Procedures (SOPs) indexed in the Pinecone Vector DB.
    Use this to retrieve regulatory thresholds, cold-chain breach mitigations, and rerouting rules.
    """
    try:
        # Pesquisa no índice vetorial os trechos mais relacionados à pergunta recebida.
        # Caminho interno: query -> embeddings -> busca no Pinecone -> Documents.
        # Cada Document contém page_content (trecho) e metadata (informações de origem).
        # k=2 foi definido na inicialização; não há um limiar de relevância neste código.
        matched_docs = retriever.invoke(query)
        # Se a busca não devolver documentos, retorna uma mensagem de ausência de contexto.
        if not matched_docs:
            return "No matching compliance clauses found."
            
        # Combina os trechos encontrados; inclui metadados para identificar a origem de cada trecho.
        # A compreensão de lista cria um texto por documento; join os une com uma
        # linha em branco. get usa SOP/RAW como padrão se o metadado não existir.
        formatted_context = "\n\n".join(
            [f"[Source: {doc.metadata.get('source_file', 'SOP')} | Format: {doc.metadata.get('file_format', 'RAW')}]\n{doc.page_content}" for doc in matched_docs]
        )
        # Devolve o contexto formatado para que o agente possa usá-lo na resposta.
        return f"--- COMPLIANCE SOP CONTEXT ---\n{formatted_context}\n------------------------------"
    # Se a consulta ao índice vetorial falhar, devolve uma mensagem com o erro.
    except Exception as e:
        return f"Vector Store Retrieval Error: {str(e)}"
    
# ==========================================
# 3. VERIFICAÇÃO DIRETA DAS FERRAMENTAS PELO TERMINAL
# ==========================================
# Este bloco só roda quando agent_tools.py é executado diretamente.
# Se outro arquivo importar este módulo, os testes abaixo não serão iniciados automaticamente.
# Aqui, .invoke(...) executa diretamente o objeto criado por @tool: não há LLM,
# grafo, conversa ou interface escolhendo o que chamar. São chamadas reais aos serviços.
# Uma ferramenta com um argumento textual pode receber uma string; a ferramenta
# com latitude e longitude recebe um dicionário com os nomes dos dois argumentos.
if __name__ == "__main__":
    # Mostra no terminal que o teste da consulta ao banco está começando.
    print("\n--- Testing Tool 1: SQL Telemetry View ---")
    # Chama a ferramenta SQL com uma consulta de exemplo e exibe o resultado.
    print(query_telemetry_db.invoke("SELECT TOP 2 Latitude, Longitude, Current_Temperature_C FROM FDE_VIEWS.VW_ACTIVE_FLEET"))
    
    # Mostra no terminal que o teste da consulta de clima está começando.
    print("\n--- Testing Tool 2: Live Corridor API ---")
    # Envia latitude e longitude à ferramenta e exibe as condições do corredor.
    print(fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19}))
    
    # Mostra no terminal que o teste de busca nos procedimentos está começando.
    print("\n--- Testing Tool 3: Pinecone Vector Retrieval ---")
    # Pesquisa no índice por regras de temperatura e exibe os trechos encontrados.
    print(search_compliance_sop.invoke("What are the temperature rules for fresh perishables?"))
