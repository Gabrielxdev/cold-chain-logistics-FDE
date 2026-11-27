# GUIA DE LEITURA
# Este arquivo cria a interface web do agente usando Streamlit.
# Fluxo principal: pergunta -> grafo do agente -> ferramentas -> resposta e logs.
# O Streamlit executa este script de cima para baixo e volta a executá-lo quando
# há interações com os widgets. Por isso, dados que precisam sobreviver a essas
# execuções ficam em st.session_state, como o histórico exibido no chat.
# A função definida aqui é write_audit_log; os demais blocos montam a página
# e reagem às ações do usuário durante cada execução do script.

# os lê variáveis de ambiente; sys permite ajustar onde o Python busca módulos.
import os
import sys
# uuid gera identificadores de conversa; json transforma dados em texto JSON.
import uuid
import json
# urllib.parse codifica a string de conexão para incluí-la em uma URL.
import urllib
# Path facilita montar caminhos sem depender de barras específicas do sistema.
from pathlib import Path
# load_dotenv disponibiliza as configurações do arquivo .env no ambiente.
from dotenv import load_dotenv
# Os apelidos st e pd abreviam as chamadas ao Streamlit e ao pandas.
import streamlit as st
# pandas representa o resultado da consulta de auditoria como uma tabela (DataFrame).
import pandas as pd
# create_engine prepara o acesso ao banco; text representa SQL com parâmetros.
from sqlalchemy import create_engine, text
# HumanMessage representa a pergunta; ToolMessage representa a saída de uma ferramenta.
from langchain_core.messages import HumanMessage, ToolMessage

# ==========================================
# 1. LOCALIZAÇÃO DO PROJETO E CARREGAMENTO DO AMBIENTE
# ==========================================
# __file__ é o caminho deste arquivo; resolve() o torna absoluto e parent sobe um nível.
script_dir = Path(__file__).resolve().parent  # Diretório src/.
project_root = script_dir.parent              # Raiz do projeto, onde fica o .env.

# Inclui a raiz na busca de módulos para permitir imports como src.orchestrator.
# A verificação evita adicionar o mesmo caminho novamente a cada execução.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Por padrão, load_dotenv não substitui variáveis já definidas no ambiente.
load_dotenv(project_root / ".env")

# Importa o grafo já compilado, que coordena o modelo e as ferramentas.
# O orquestrador carrega system_prompt.txt e inclui suas instruções em cada
# chamada ao modelo, tanto nesta interface quanto na execução pelo terminal.
from src.orchestrator import fde_agent

# ==========================================
# 2. CONEXÃO SQL E FUNÇÃO DE AUDITORIA
# ==========================================
# getenv(nome, padrão) usa o padrão se a variável não existir.
# A senha não tem padrão: se estiver ausente, o valor recebido será None.
db_host = os.getenv("SQL_SERVER_HOST", "localhost")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
db_password = os.getenv("SQL_AGENT_PASSWORD")

# Monta os parâmetros ODBC para gravar logs usando o usuário do agente.
# DRIVER escolhe o driver instalado; SERVER combina host e porta; DATABASE escolhe
# o banco; UID/PWD são as credenciais. As f-strings inserem valores entre chaves;
# {{ e }} produzem chaves literais ao redor do nome do driver.
# Encrypt=no desabilita a solicitação de criptografia por esta configuração;
# TrustServerCertificate=yes dispensa a validação do certificado quando aplicável.
connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE=master;"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
)

# Escapa caracteres para transportar a string ODBC como parâmetro de URL.
# Essa codificação não criptografa nem esconde a senha.
log_params = urllib.parse.quote_plus(connection_string)

# O engine gerencia conexões via SQLAlchemy + pyodbc. A conexão efetiva é aberta
# quando connect() é chamado, não apenas ao criar este objeto.
log_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={log_params}")

# Registra uma etapa da execução na tabela de auditoria.
# Parâmetros:
# - session_id: identifica a conversa à qual o registro pertence;
# - node_name: identifica a etapa do grafo (ou o rótulo usado para o texto gerado);
# - tool_name: nome da ferramenta ou descrição da operação;
# - content: texto a armazenar, como argumentos em JSON ou resultado da ferramenta.
# Não retorna um resultado explícito: o retorno é None, inclusive em caso de falha.
def write_audit_log(session_id, node_name, tool_name, content):
    """Silently writes agent execution traces to the SQL audit table using agent permissions."""
    # try/except impede que uma falha de auditoria interrompa o atendimento.
    try:
        # O with libera a conexão ao sair do bloco, mesmo se ocorrer uma exceção.
        with log_engine.connect() as conn:
            # :session_id e os demais marcadores recebem valores do dicionário abaixo.
            # Assim, o conteúdo é enviado como dado, sem concatená-lo ao comando SQL.
            # FDE_VIEWS é o schema e AgentAuditLog é a tabela dentro do banco master.
            conn.execute(text("""
                INSERT INTO FDE_VIEWS.AgentAuditLog (SessionID, NodeExecuted, ToolName, Content)
                VALUES (:session_id, :node_name, :tool_name, :content)
            """), {
                "session_id": session_id,
                "node_name": node_name,
                "tool_name": tool_name,
                "content": content
            })
            # Confirma a transação para persistir a inserção no banco.
            conn.commit()
    except Exception as e:
        # O erro aparece no terminal do servidor, sem um alerta na interface.
        print(f"Audit Log Failed (Silent): {e}")

# ==========================================
# 3. CONFIGURAÇÃO DA PÁGINA E DO TEMA VISUAL
# ==========================================
# Define título/ícone da aba do navegador, largura ampla e barra lateral aberta.
st.set_page_config(
    page_title="FDE Supply Chain Dispatch Console",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Insere CSS para definir cores do fundo, da barra lateral e dos trechos de código.
# unsafe_allow_html=True permite interpretar o HTML <style> desta string fixa.
# Os seletores identificam elementos da página; !important dá prioridade à regra.
st.markdown("""
<style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    div[data-testid="stSidebar"] { background-color: #111622; border-right: 1px solid #1E293B; }
    .stMarkdown code { background-color: #1E293B !important; color: #38BDF8 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. ESTADO DA INTERFACE E IDENTIFICAÇÃO DA CONVERSA
# ==========================================
# Streamlit reexecuta este arquivo quando o usuário interage com a página.
# session_state conserva valores durante essas reexecuções da mesma sessão,
# então podemos guardar nele um identificador para esta conversa.
# A condição pergunta: "já existe um thread_id nesta sessão?"
# Se não existir, uuid4() cria um identificador único e str() o converte em texto.
# Se já existir, não o substituímos: assim o LangGraph continua usando a mesma
# conversa (e a mesma memória) em vez de começar uma conversa nova a cada interação.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# A interface também precisa guardar as mensagens para conseguir desenhá-las no chat.
# Esta condição verifica se o histórico "ui_messages" já foi criado nesta sessão.
# Na primeira execução, ainda não há mensagens, então começamos com uma lista vazia.
# Nas interações seguintes, a lista existente é mantida e recebe as novas mensagens;
# ela não é apagada a cada reexecução do arquivo pelo Streamlit.
# Cada item da lista representa uma mensagem: role indica quem falou, content guarda
# o texto e, nas respostas do agente, traces pode guardar detalhes das ferramentas.
# Esta lista é para exibir o chat na tela; o histórico usado pelo modelo é mantido
# separadamente pelo estado/checkpointer do LangGraph.
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# Monta as opções que serão passadas ao LangGraph quando a UI executar o agente.
# "configurable" é o espaço de configuração usado pelo LangGraph; nele, "thread_id"
# identifica qual conversa o checkpointer deve recuperar ou atualizar.
# Aqui usamos o ID guardado no session_state: todas as chamadas desta sessão recebem
# o mesmo identificador e, por isso, continuam o mesmo histórico. Outro ID representa
# outra conversa. Este dicionário não contém mensagens: ele identifica onde o LangGraph
# deve buscar/salvar o estado. As mensagens mostradas na tela ficam em ui_messages.
# No orchestrator.py, o checkpointer é MemorySaver, que guarda o estado na memória
# do processo; os registros de auditoria, por sua vez, ficam no SQL Server.
thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ==========================================
# 5. BARRA LATERAL, NAVEGAÇÃO E REINÍCIO DA CONVERSA
# ==========================================
# Tudo dentro deste bloco aparece na barra lateral, separada da área principal.
with st.sidebar:
    # Mostra a imagem local quando disponível; se não, usa a imagem da URL.
    st.image(str(script_dir / "image_L25X5q.png") if (script_dir / "image_L25X5q.png").exists() else "https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=65)
    st.title("FDE Command Center")
    
    # O usuário escolhe qual tela quer abrir. O texto selecionado fica em app_mode
    # e mais abaixo decide se mostramos o chat ou os registros de auditoria.
    app_mode = st.radio("System Mode", ["🧊 Dispatch Console", "🛡️ Security & Audit Logs"])
    
    st.markdown("---")
    # Mostra uma versão curta do ID da conversa. [:8] exibe só os 8 primeiros
    # caracteres; o ID completo continua guardado em st.session_state.
    st.caption(f"Session Token: `{st.session_state.thread_id[:8]}...`")
    # Apenas exibe o provedor configurado no ambiente; esta linha não escolhe
    # nem altera o modelo. A escolha efetiva é feita em orchestrator.py.
    st.markdown(f"**Reasoning Architecture:** `{os.getenv('Agent_llm', 'DEEPSEEK')}`")
    
    st.markdown("---")
    # Reinicia a conversa visível e troca seu identificador. O novo thread_id fará
    # com que as próximas chamadas ao LangGraph usem um estado separado.
    # Isso não apaga os registros de auditoria do SQL nem os checkpoints antigos.
    if st.button("🗑️ Purge Dispatch Workspace Session", use_container_width=True):
        st.session_state.ui_messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        # Faz o Streamlit executar o arquivo novamente agora, para que thread_config
        # e a tela sejam reconstruídos com o ID novo.
        st.rerun()

# ==========================================
# 6. ESCOLHA DA TELA: CHAT OPERACIONAL OU CONSULTA DE AUDITORIA
# ==========================================

# app_mode escolhe exatamente um dos dois caminhos abaixo: chat operacional
# ou consulta de auditoria. Os componentes da outra tela não são executados.
if app_mode == "🧊 Dispatch Console":
    # ------------------------------------------
    # TELA 1: HISTÓRICO DO CHAT E EXECUÇÃO DO AGENTE
    # ------------------------------------------
    st.title("Cold-Chain Incident Control Dashboard (Project FDE)")
    st.caption("Production Data Engineering Pipeline • Real-Time Decision Optimization Platform")

    # Streamlit reexecuta este arquivo após interações. Primeiro redesenhamos
    # as mensagens que já estão em ui_messages; isso só atualiza a tela e não
    # envia essas mensagens novamente ao modelo.
    for entry in st.session_state.ui_messages:
        # Abre um balão de chat. role diz se a mensagem é do usuário ou do agente;
        # o avatar é escolhido de acordo com esse autor.
        with st.chat_message(entry["role"], avatar="👤" if entry["role"] == "user" else "🤖"):
            # Respostas do agente podem guardar traces: detalhes das ferramentas
            # usados para reconstruir o que aconteceu. Mensagens do usuário não têm isso.
            if "traces" in entry:
                for trace in entry["traces"]:
                    # Mostra qual ferramenta o modelo pediu e os argumentos enviados a ela.
                    if trace["type"] == "tool_input":
                        st.markdown(f"**⚡ Intent Recognized:** `{trace['name']}`")
                        with st.expander(f"📥 View Generated Input ({trace['name']})", expanded=False):
                            st.json(trace["args"])
                    # Mostra o resultado bruto devolvido pela ferramenta em uma área recolhível.
                    elif trace["type"] == "tool_output":
                        with st.expander(f"📤 View Raw Output ({trace['name']})", expanded=False):
                            st.code(trace["content"], language="text")
            # Depois dos detalhes, mostra o texto da mensagem no balão.
            st.markdown(entry["content"])

    # Aguarda uma nova pergunta. chat_input retorna vazio enquanto nada foi enviado;
    # quando o usuário envia texto, := guarda esse texto em user_input e entra no bloco.
    if user_input := st.chat_input("Query fleet telemetry, corridor updates, or compliance thresholds..."):
        
        # Salva a pergunta no histórico visual para ela continuar aparecendo após
        # futuras reexecuções do Streamlit e mostra-a imediatamente na tela.
        st.session_state.ui_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Abre o balão do agente. Tudo que ocorrer nesta execução — progresso,
        # chamadas/resultados de ferramentas e resposta final — será mostrado aqui.
        with st.chat_message("assistant", avatar="🤖"):
            # São variáveis temporárias para esta pergunta: texto final do modelo
            # e detalhes das ferramentas que depois serão guardados no histórico visual.
            final_response = ""
            current_traces = [] 
            
            # Exibe o andamento enquanto o grafo trabalha. O rótulo muda conforme
            # chegam atualizações dos nós do LangGraph.
            with st.status("🧠 Initializing Core Reasoner Node...", expanded=True) as status:
                # Envia somente a pergunta nova. O checkpointer encontra o histórico
                # anterior pelo thread_id dentro de thread_config; ui_messages é apenas
                # o histórico visual e não é enviado ao agente.
                # stream_mode="updates" permite acompanhar o que cada nó atualiza
                # (reasoner/tools), em vez de receber o texto token por token.
                events = fde_agent.stream(
                    {"messages": [HumanMessage(content=user_input)]}, 
                    config=thread_config,
                    stream_mode="updates"
                )
                
                # Lê os eventos conforme o grafo avança. O ciclo possível é:
                # reasoner pede uma ferramenta -> tools executa -> reasoner lê o resultado.
                for event in events:
                    # Um evento informa qual nó terminou e quais atualizações ele produziu.
                    for node_name, node_state in event.items():
                        
                        if node_name == "reasoner":
                            # O nó reasoner pode produzir uma mensagem da IA; [-1] pega
                            # a mais recente desta atualização.
                            latest_msg = node_state["messages"][-1]
                            
                            # Se o modelo pediu ferramentas, exibe cada pedido e seus
                            # argumentos. Aqui só mostramos/registramos o pedido; o nó
                            # "tools" do grafo é quem realmente executa as ferramentas.
                            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                                status.update(label="🧠 Agent generated tool parameters...")
                                # Uma única resposta da IA pode pedir mais de uma ferramenta.
                                for tool_call in latest_msg.tool_calls:
                                    st.markdown(f"**⚡ Intent Recognized:** `{tool_call['name']}`")
                                    with st.expander(f"📥 View Generated Input ({tool_call['name']})", expanded=False):
                                        st.json(tool_call['args'])
                                    
                                    # Guarda pedido e argumentos para poder redesenhá-los
                                    # no histórico visual depois que a execução terminar.
                                    current_traces.append({
                                        "type": "tool_input",
                                        "name": tool_call['name'],
                                        "args": tool_call['args']
                                    })
                                    
                                    # Registra também o pedido na auditoria do banco.
                                    # json.dumps transforma os argumentos (dicionário)
                                    # em texto JSON para serem armazenados no campo Content.
                                    write_audit_log(
                                        session_id=st.session_state.thread_id,
                                        node_name="reasoner",
                                        tool_name=tool_call['name'],
                                        content=json.dumps(tool_call['args'])
                                    )
                            
                            # A mesma mensagem da IA também pode conter texto. Se houver,
                            # guardamos esse texto como resposta mais recente. Este teste
                            # é separado do de tool_calls: o modelo pode fornecer texto
                            # e pedir uma ferramenta na mesma mensagem.
                            if latest_msg.content:
                                final_response = latest_msg.content
                                status.update(label="📝 Generating Operational Resolution Report...")
                                
                                # Salva esse texto na auditoria. Uma mensagem com texto
                                # não significa necessariamente que o grafo terminou;
                                # após uma ferramenta, o reasoner pode produzir outro texto.
                                write_audit_log(
                                    session_id=st.session_state.thread_id,
                                    node_name="reasoner_final",
                                    tool_name="LLM Text Synthesis",
                                    content=final_response
                                )
                                
                        # Este evento chega depois que o nó tools executou os pedidos.
                        elif node_name == "tools":
                            status.update(label="🔧 Executing Enterprise Subsystem Tools...")
                            # O nó tools devolve mensagens com os resultados. Percorremos
                            # a lista messages; se ela não existir, não há resultados a ler.
                            for msg in node_state.get("messages", []):
                                # ToolMessage representa a resposta de uma ferramenta,
                                # então ignoramos qualquer outro tipo de mensagem.
                                if isinstance(msg, ToolMessage):
                                    with st.expander(f"📤 View Raw Output ({msg.name})", expanded=False):
                                        st.code(msg.content, language="text")
                                        
                                    # Guarda o resultado para exibi-lo novamente no chat
                                    # e registra uma cópia na tabela de auditoria do SQL Server.
                                    current_traces.append({
                                        "type": "tool_output",
                                        "name": msg.name,
                                        "content": msg.content
                                    })
                                    
                                    write_audit_log(
                                        session_id=st.session_state.thread_id,
                                        node_name="tools",
                                        tool_name=msg.name,
                                        content=msg.content
                                    )
                
                # O fluxo terminou e todos os eventos foram processados; marca o painel
                # como concluído e o recolhe para deixar a resposta em destaque.
                status.update(label="Incident Matrix Evaluation Complete", state="complete", expanded=False)
                
            # Depois do fim do grafo, mostra a última resposta de texto e salva no
            # histórico visual a resposta junto com os traces desta pergunta.
            if final_response:
                st.markdown(final_response)
                st.session_state.ui_messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "traces": current_traces
                })
            else:
                # Se nenhum texto foi recebido, mostra uma mensagem alternativa.
                # Apesar de dizer "Timeout", este trecho não mede tempo nem captura
                # exceções; apenas indica que não houve resposta textual.
                error_fallback = "⚠️ Execution Timeout: System engine encountered an unresolved processing edge case."
                st.error(error_fallback)


elif app_mode == "🛡️ Security & Audit Logs":
    # Este modo é um painel de consulta, não uma conversa com o agente.
    # Ele permite que um administrador inspecione no SQL Server os eventos
    # registrados durante as conversas: horário, sessão, etapa do grafo,
    # ferramenta chamada e conteúdo gravado. Serve para acompanhar e investigar
    # a atividade do agente; a consulta abaixo pode mostrar registros de várias
    # conversas e não altera nem reinicia o estado do agente.
    # ------------------------------------------
    # TELA 2: CONSULTA DOS LOGS COM AS CREDENCIAIS ADMINISTRATIVAS CONFIGURADAS
    # ------------------------------------------
    st.title("🛡️ Enterprise Agent Audit Trail")
    st.caption("Secure database inspection of FDE_VIEWS.AgentAuditLog")
    
    st.markdown("### Database Authorization Gate")
    st.markdown("Enter high-privilege administrative credentials (defined in `.env` as `SQL_ADMIN_USER`) to query audit logs.")
    
    # Agrupa usuário, senha e botão. Os valores só são enviados juntos quando
    # o botão do formulário é clicado.
    with st.form("admin_auth_form"):
        # Organiza os campos lado a lado: usuário na primeira coluna, senha na segunda.
        col1, col2 = st.columns(2)
        with col1:
            # Usa o usuário configurado no ambiente como valor inicial; ele pode ser editado.
            input_user = st.text_input("Admin Username", value=os.getenv("SQL_ADMIN_USER", ""))
        with col2:
            # Mascara o que é digitado e não pré-carrega a senha na tela.
            input_pass = st.text_input("Admin Password", type="password", value="")
            
        # Depois do clique, submit_admin será True nesta execução do script.
        submit_admin = st.form_submit_button("Authenticate & Load Logs", use_container_width=True)

    # Só tenta autenticar e consultar o banco quando o formulário foi enviado.
    # Este código não mantém um estado de login para as execuções futuras.
    if submit_admin:
        expected_admin_user = os.getenv("SQL_ADMIN_USER")
        expected_admin_pass = os.getenv("SQL_ADMIN_PASSWORD")
        
        # Primeiro compara o que foi digitado com os valores esperados no ambiente.
        # Se coincidirem, ainda será necessário que o SQL Server aceite a conexão
        # e autorize a leitura da tabela.
        if input_user == expected_admin_user and input_pass == expected_admin_pass:
            try:
                # Monta uma conexão de leitura com as credenciais administrativas.
                # É um engine separado daquele usado pelo agente para gravar auditoria.
                admin_params = urllib.parse.quote_plus(
                    "DRIVER={ODBC Driver 18 for SQL Server};"
                    f"SERVER={db_host},{db_port};"
                    "DATABASE=master;"
                    f"UID={input_user};"
                    f"PWD={input_pass};"
                    "Encrypt=no;"
                    "TrustServerCertificate=yes;"
                )
                admin_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={admin_params}")
                
                # Abre a conexão só durante este bloco. O DataFrame com os resultados
                # continua disponível depois que a conexão é fechada.
                with admin_engine.connect() as conn:
                    # Busca todos os registros de auditoria, do mais novo para o mais antigo.
                    # Como não há filtro por SessionID, podem aparecer várias conversas.
                    query = """
                        SELECT LogID, Timestamp, SessionID, NodeExecuted, ToolName, Content 
                        FROM FDE_VIEWS.AgentAuditLog 
                        ORDER BY Timestamp DESC
                    """
                    # Executa o SELECT e coloca as linhas retornadas em uma tabela pandas.
                    df = pd.read_sql(query, conn)
                
                st.success("✅ Authenticated successfully as Admin.")
                
                # Se houver linhas, apresenta-as em uma tabela; se não, informa que
                # a consulta funcionou, mas ainda não há registros.
                if not df.empty:
                    # Ajusta os nomes e a apresentação das colunas da tabela na tela.
                    st.dataframe(
                        df,
                        column_config={
                            "LogID": st.column_config.NumberColumn("ID", format="%d"),
                            "Timestamp": st.column_config.DatetimeColumn("Execution Time", format="DD/MM/YYYY-h:mm a"),
                            "SessionID": "Session Token",
                            "NodeExecuted": "Graph Node",
                            "ToolName": "Tool Triggered",
                            "Content": "Raw Payload Data"
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=600
                    )
                else:
                    st.info("No audit logs found in the database. Run a query in the Dispatch Console first.")
                    
            except Exception as e:
                # Mostra na tela qualquer erro de conexão ou de execução do SELECT.
                st.error(f"Database Query Failed: {e}")
        else:
            # Credenciais diferentes das configuradas: não tenta conectar ao banco.
            st.error("❌ Invalid Administrator Credentials.")
