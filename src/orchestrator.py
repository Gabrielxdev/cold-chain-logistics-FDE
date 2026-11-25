# MAPA DO FLUXO — ESTE ARQUIVO CONECTA ui.py A agent_tools.py
#
# ui.py: pergunta -> HumanMessage -> fde_agent.stream(..., config=thread_config)
#                                      |
#                                      v
#                                   START
#                                      |
#                                      v
#                         reasoner = reasoning_node
#                         envia state["messages"] ao LLM
#                                      |
#                               tools_condition
#                         /                         \
#                  há tool_calls               não há tool_calls
#                       |                            |
#                       v                            v
#                 tools = ToolNode                  END
#                 executa agent_tools.py
#                       |
#                 retorna ToolMessage
#                       |
#                       +-----------> reasoner (o ciclo pode se repetir)
#
# ui.py acompanha as atualizações de reasoner/tools para mostrar progresso e logs.
# A ordem das ferramentas não é fixa: o modelo pode chamar nenhuma, uma ou várias.
# END encerra esta execução do grafo; o estado da conversa pode ser usado na próxima
# pergunta se ela vier com o mesmo thread_id e o checkpointer ainda tiver o histórico.
#
# Há dois momentos de leitura: primeiro a preparação de modelo/ferramentas/grafo;
# depois a execução desse grafo, disparada pela interface ou pelo bloco de terminal.

# Acessa variáveis de ambiente e altera a lista de caminhos onde o Python procura módulos.
import os
import sys
# Resolve e manipula caminhos de arquivos de forma compatível com o sistema operacional.
from pathlib import Path

# Lê configurações de um arquivo .env e disponibiliza os valores em variáveis de ambiente.
from dotenv import load_dotenv

# Tipos usados para descrever os dados compartilhados pelas etapas do agente.
from typing import Annotated, TypedDict

# Tipos de mensagem usados para representar a conversa e as instruções do sistema.
from langchain_core.messages import BaseMessage, SystemMessage

# Componentes do LangGraph: StateGraph monta o fluxo; START e END representam seus limites.
from langgraph.graph import StateGraph, START, END

# Define como as mensagens novas são adicionadas às que já estão no estado.
from langgraph.graph.message import add_messages

# ToolNode executa as ferramentas; tools_condition decide se o fluxo deve chamá-las.
from langgraph.prebuilt import ToolNode, tools_condition

# Guarda o estado das conversas em memória enquanto este processo estiver ativo.
from langgraph.checkpoint.memory import MemorySaver



# ==========================================
# 1. PREPARAÇÃO E LOCALIZAÇÃO DOS CAMINHOS
# ==========================================
# Obtém o diretório deste arquivo, já convertido para um caminho absoluto.
script_dir = Path(__file__).resolve().parent
# Neste projeto, o diretório pai de src é a raiz do projeto.
project_root = script_dir.parents[0]

# Carrega as instruções também quando ui.py importa este módulo.
# Reinicie o processo após editar o arquivo para carregar a nova versão.
prompt_path = project_root / "src" / "prompts" / "system_prompt.txt"
system_instructions = prompt_path.read_text(encoding="utf-8").strip()
if not system_instructions:
    raise ValueError(f"O prompt de sistema está vazio: {prompt_path}")
system_prompt = SystemMessage(content=system_instructions)

# Verifica se a raiz do projeto já está nos caminhos de busca de módulos do Python.
if str(project_root) not in sys.path:
    # Coloca a raiz no início da lista para que os módulos do projeto sejam encontrados.
    sys.path.insert(0, str(project_root))

# Importa as ferramentas que o modelo poderá usar para consultar dados de logística.
# Esta importação carrega a configuração de agent_tools.py e prepara embeddings
# e retriever. Os corpos das três ferramentas só executam quando elas são chamadas.
# Na primeira carga pela UI: ui.py importa este módulo, que importa agent_tools.py;
# depois a execução volta aqui para montar o grafo e, por fim, volta à interface.
from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop

# Carrega as configurações do arquivo .env localizado na raiz do projeto.
load_dotenv(project_root / ".env")

# Define o formato do "estado": um dicionário que os nós do grafo recebem e atualizam.
# TypedDict descreve as chaves esperadas para leitura/tipagem; não cria uma conversa
# nem valida sozinho cada valor em tempo de execução. O grafo gerencia o estado.
class AgentState(TypedDict):
    # O estado é um dicionário mantido pelo LangGraph. A chave "messages" guarda
    # o histórico que o modelo recebe para entender o que já aconteceu na conversa.
    #
    # list[BaseMessage] significa "uma lista de mensagens do LangChain".
    # BaseMessage é o tipo geral; os itens concretos podem ser, por exemplo,
    # SystemMessage (instruções), HumanMessage (usuário), AIMessage (modelo)
    # ou ToolMessage (resultado de uma ferramenta).
    #
    # Annotated adiciona uma regra que o LangGraph usa ao atualizar esse campo:
    # add_messages combina as mensagens novas com o histórico existente.
    # Exemplo: se o histórico for [SystemMessage, HumanMessage] e um nó devolver
    # {"messages": [AIMessage]}, o resultado será [SystemMessage, HumanMessage,
    # AIMessage] — não apenas [AIMessage]. Sem essa regra, a lista anterior
    # poderia ser substituída pela atualização do nó.
    #
    # Quando o modelo pede uma ferramenta, o ciclo normalmente acrescenta:
    # AIMessage (pedido da ferramenta) -> ToolMessage (resultado) ->
    # AIMessage (resposta ao usuário). Assim, na próxima chamada, o modelo
    # recebe tanto a pergunta quanto o pedido e o resultado da ferramenta.
    # add_messages também pode substituir uma mensagem se o ID dela já existir;
    # neste fluxo normal, cada nova mensagem é acrescentada ao histórico.
    messages: Annotated[list[BaseMessage], add_messages]

# ==========================================
# 2. CONFIGURAÇÃO DO MODELO DE LINGUAGEM
# ==========================================

# Lê qual provedor usar; se não houver configuração, escolhe OLLAMA.
# strip remove espaços externos e upper normaliza o texto para maiúsculas.
# Agent_llm controla quem interpreta a conversa; Embeddings_model, em agent_tools.py,
# controla quem vetoriza textos para buscar documentos. São escolhas independentes.
AGENT_LLM_SETTING = os.getenv("Agent_llm", "OLLAMA").strip().upper()

# Se a configuração pedir OPENAI, prepara o cliente correspondente.
if AGENT_LLM_SETTING == "OPENAI":
    print("🤖 Brain Mode: Utilizing Cloud OpenAI Reasoner (gpt-4o)...")
    # Importa o cliente OpenAI somente quando essa opção foi selecionada.
    from langchain_openai import ChatOpenAI
    # Cria o modelo; temperatura zero reduz a variação das respostas.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Caso contrário, se a configuração pedir DEEPSEEK, usa o cliente compatível com sua API.
elif AGENT_LLM_SETTING == "DEEPSEEK":
    print("🐳 Brain Mode: Utilizing Flagship DeepSeek Cloud Reasoner (deepseek-v4-pro)...")
    # O cliente OpenAI também pode se conectar à API DeepSeek usando outra URL-base.
    from langchain_openai import ChatOpenAI
    
    # Configura o modelo DeepSeek e os parâmetros da conexão com o serviço.
    llm = ChatOpenAI(
        model="deepseek-v4-flash",                           # Nome do modelo DeepSeek solicitado à API.
        temperature=0,                                       # Mantém as respostas menos aleatórias.
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),         # Lê a chave da API do ambiente/.env.
        base_url="https://api.deepseek.com",                  # Endereço-base da API DeepSeek.
        max_tokens=2048,                                      # Limita o tamanho máximo da resposta gerada.
        # Os parâmetros abaixo estão desativados; só seriam enviados se fossem descomentados.
        # extra_body={
        #     "thinking": {"type": "enabled"},              # Solicita o modo de raciocínio ampliado.
        #     "reasoning_effort": "high"                     # Solicita alto esforço de raciocínio.
        # }
    )

# Para qualquer outro valor, usa Ollama local como opção padrão/de reserva.
else:
    print("🤗 Brain Mode: Local Fallback Activated. Binding Local Ollama (qwen2.5:7b)...")
    # Importa o cliente local do Ollama apenas para esta opção.
    from langchain_community.chat_models import ChatOllama
    # Define o modelo local, com baixa aleatoriedade e limite de tokens de saída.
    llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_predict=1024)

# PONTE COM agent_tools.py: reúne os objetos criados pelo decorador @tool.
# Esta mesma lista terá dois usos: informar ao modelo o que ele pode solicitar
# e informar ao executor ToolNode quais implementações estão disponíveis.
fde_tools = [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]
# Vincula essas ferramentas ao modelo para que ele possa emitir solicitações para executá-las.
# bind_tools fornece nomes, descrições e formatos de argumentos; não executa SQL,
# HTTP ou buscas. O modelo devolve tool_calls e o grafo cuida da execução depois.
llm_with_tools = llm.bind_tools(fde_tools)

# ==========================================
# 3. MONTAGEM DO FLUXO DE EXECUÇÃO
# ==========================================
# Nó do grafo responsável pela etapa de raciocínio: recebe o estado atual da conversa,
# pergunta ao modelo o que fazer e devolve a resposta para atualizar o estado.
# Entrada: AgentState com o histórico acumulado. Saída: atualização contendo uma
# mensagem do modelo (AIMessage). Na UI, o nó aparece sob a chave "reasoner".
def reasoning_node(state: AgentState):
    # `state` é o dicionário do grafo; `state["messages"]` é a lista com o histórico da conversa.
    # `invoke(...)` envia esse histórico ao modelo e espera até ele produzir uma resposta.
    # Como as ferramentas foram vinculadas ao modelo, essa resposta pode ser:
    # 1. texto destinado ao usuário; ou
    # 2. uma solicitação estruturada para executar uma ferramenta.
    # Na primeira passagem, lê a pergunta; após tools, lê também os resultados.
    # Uma AIMessage pode ter content (texto) e tool_calls ao mesmo tempo.
    # Cada tool_call traz name, args e id; o grafo usa essas informações para executar
    # a ferramenta correta e associar o resultado à solicitação correspondente.
    # Inclui as instruções em toda chamada, inclusive após resultados de ferramentas.
    # A lista é nova: o prompt não se acumula no histórico salvo pelo checkpointer.
    messages = [system_prompt, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    # O nó devolve as mudanças no estado no formato {nome_do_campo: novo_valor}.
    # Aqui, `response` é uma única mensagem nova. O `add_messages` definido em AgentState
    # faz com que ela seja acrescentada ao histórico existente, sem substituir as mensagens anteriores.
    # Se for uma solicitação de ferramenta, o grafo encaminha essa mensagem ao nó `tools`;
    # se for uma resposta final, o fluxo pode terminar.
    return {"messages": [response]}

# Informa no terminal que o fluxo está sendo montado.
print("⚙️ Compiling LangGraph FDE Orchestrator...")
# Cria o "rascunho" do fluxo do agente. StateGraph recebe AgentState para saber
# qual formato de estado os nós vão ler e atualizar (neste caso, o dicionário
# com a chave "messages" e a regra add_messages).
# Neste ponto, o agente ainda não executa nada: graph_builder é só o montador.
# As linhas add_node e add_edge abaixo registram as etapas e suas conexões;
# mais adiante, compile() transforma essa definição em um grafo executável.
graph_builder = StateGraph(AgentState)
# Registra o nó que chama o modelo de linguagem.
# "reasoner" é o nome nos eventos; reasoning_node é a função Python executada.
graph_builder.add_node("reasoner", reasoning_node)
# Registra um nó que executa as ferramentas solicitadas pelo modelo.
# ToolNode lê tool_calls, encontra cada ferramenta de fde_tools pelo nome e passa
# seus args para ela. O retorno textual vira uma ToolMessage com o tool_call_id
# correspondente. Assim, o LLM sabe qual solicitação cada resultado respondeu.
# A UI reconhece ToolMessage e exibe seu content no painel de saída bruta.
graph_builder.add_node("tools", ToolNode(fde_tools))

# Pense no grafo como um mapa de etapas: esta linha desenha a seta
# "início da execução -> etapa de raciocínio".
# START é um marcador especial do LangGraph (não é uma função nossa);
# "reasoner" é o nome dado ao nó registrado acima, que executa reasoning_node.
# Assim, ao iniciar uma conversa, o LangGraph entrega o estado atual
# (incluindo "messages") a reasoning_node para o modelo decidir o próximo passo.
# Esta linha apenas desenha a seta no mapa; ela não chama o modelo por si só.
graph_builder.add_edge(START, "reasoner")
# Registra uma bifurcação no mapa, que será avaliada depois que "reasoner" rodar.
# tools_condition olha a mensagem mais recente do modelo:
# - se ela pedir uma ferramenta (tool_calls), o fluxo segue para o nó "tools";
# - se não pedir ferramenta, o modelo já respondeu e o fluxo segue para END.
# Exemplo: "consulte a temperatura" pode gerar tool_calls e levar a "tools";
# "a temperatura é 5 graus" não pede ferramenta e encerra o fluxo.
# Importante: tools_condition só decide o caminho. A ferramenta e seus argumentos
# são escolhidos pelo modelo e executados pelo nó "tools", não por esta linha.
graph_builder.add_conditional_edges("reasoner", tools_condition)
# Desenha a seta de retorno: depois que "tools" executa a ferramenta solicitada,
# o resultado é acrescentado ao histórico como uma ToolMessage.
# O fluxo volta a "reasoner" para o modelo ler esse resultado junto com a pergunta
# original e decidir o que fazer: responder ao usuário ou pedir outra ferramenta.
# Exemplo do ciclo: reasoner (pede ferramenta) -> tools (executa e registra resultado)
# -> reasoner (interpreta o resultado). Sem esta seta, o modelo não receberia
# automaticamente a resposta da ferramenta para continuar a conversa.
graph_builder.add_edge("tools", "reasoner")

# Compila o grafo e habilita o armazenamento de estado/checkpoints em memória.
# compile produz o objeto executável que ui.py importa como fde_agent.
# MemorySaver separa os estados pelo thread_id informado em config["configurable"].
# Ele não é a tabela SQL de auditoria nem a lista ui_messages do Streamlit:
# - estado do grafo: mensagens usadas pelo modelo para continuar a conversa;
# - ui_messages: conteúdo salvo pela interface para redesenhar o chat;
# - AgentAuditLog: registros persistidos por write_audit_log, definida em ui.py.
# Rodar só este orquestrador no terminal não chama aquela função de auditoria.
fde_agent = graph_builder.compile(checkpointer=MemorySaver())

# ==========================================
# 4. EXECUÇÃO INTERATIVA PELO TERMINAL
# ==========================================
# Este bloco só roda quando o arquivo é executado diretamente, não quando é importado.
# É uma entrada alternativa à UI: usa input/print para conversar pelo terminal.
# Quando ui.py importa fde_agent, este bloco é ignorado. O prompt já foi carregado
# acima e reasoning_node o inclui nas chamadas de ambas as entradas.
if __name__ == "__main__":
    # Exibe um cabeçalho para identificar o programa no terminal.
    print("\n" + "="*55)
    print("🚀 FDE Supply Chain Orchestrator State Machine Online")
    # Mostra o provedor escolhido e o valor configurado para o modelo de embeddings.
    print(f"   Configured Execution: [LLM: {AGENT_LLM_SETTING}] -> [Embeddings: {os.getenv('Embeddings_model', 'LOCAL')}]")
    print("="*55 + "\n")
    
    # Identifica esta conversa; o checkpointer usa esse ID para manter seu estado entre chamadas.
    thread_config = {"configurable": {"thread_id": "production_test_1"}}
    # O modelo só é chamado após uma pergunta; reasoning_node acrescenta o prompt.
    
    # Mantém o terminal aguardando novas mensagens até o usuário encerrar.
    while True:
        # Lê uma mensagem digitada pelo usuário.
        user_input = input("\nDispatcher > ")
        # Permite sair do loop digitando exit ou quit, independentemente de maiúsculas/minúsculas.
        if user_input.lower() in ['exit', 'quit']:
            break
            
        # Executa o grafo e recebe atualizações à medida que cada nó termina.
        # A tupla ("user", user_input) é convertida em mensagem humana pelo fluxo;
        # a UI constrói HumanMessage explicitamente para representar a mesma entrada.
        # O mesmo thread_id recupera o histórico, por isso enviamos só a nova pergunta.
        # stream(..., stream_mode="updates") permite acompanhar atualizações por nó;
        events = fde_agent.stream({"messages": [("user", user_input)]}, config=thread_config, stream_mode="updates")
        # Cada evento associa o nome de um nó às mudanças que esse nó fez no estado.
        # Formatos ilustrativos: {"reasoner": {"messages": [AIMessage(...)]}} e
        # {"tools": {"messages": [ToolMessage(...)]}}. São atualizações, não
        # necessariamente o histórico completo. ui.py percorre a mesma estrutura.
        for event in events:
            # Percorre os nós atualizados neste evento.
            for node_name, node_state in event.items():
                # Este evento chega depois que o nó tools produziu seus resultados.
                if node_name == "tools":
                    print("   [System] 🔄 Retrieving external data elements via ToolNode...")
                # Quando o modelo responde, pega sua mensagem mais recente.
                elif node_name == "reasoner":
                    latest_msg = node_state["messages"][-1]
                    # Mostra o texto, se houver; respostas que apenas solicitam ferramentas podem não ter texto.
                    if latest_msg.content:
                        print(f"\n🤖 FDE Agent:\n{latest_msg.content}")
