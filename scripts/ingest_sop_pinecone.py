# ============================================================================
# INGESTAO DE DOCUMENTOS PARA O PINECONE — VERSAO COMENTADA PARA ESTUDO
# ============================================================================
# OBJETIVO:
# Transformar documentos de procedimentos/politicas (SOPs) em vetores numericos
# pesquisaveis por similaridade semantica. Esse e o preparo da base de um RAG:
#
#   arquivos -> extracao de texto -> chunks -> embeddings -> indice Pinecone
#
# Depois, OUTRO componente (o agente/chat) pode transformar a pergunta do usuario
# em embedding e buscar os chunks mais proximos. Este script NAO responde perguntas:
# ele apenas prepara e atualiza os dados que serao consultados posteriormente.
#
# IMPORTANTE: foram adicionados comentarios; a logica original foi preservada.
# ============================================================================

# Bibliotecas padrao do Python.
import os        # Le configuracoes do ambiente (.env sera carregado abaixo).
import hashlib   # Cria a assinatura (hash) de cada arquivo para detectar mudancas.
import json      # Le e grava o cache de hashes em um arquivo JSON.
from pathlib import Path  # Manipula caminhos de pastas/arquivos em Windows e Linux.

# Bibliotecas de terceiros.
from dotenv import load_dotenv  # Carrega as variaveis do arquivo .env.
import pandas as pd             # Abre tabelas CSV e Excel.
import pypdf                    # Extrai texto de paginas de PDFs textuais.

# Pinecone: banco de dados vetorial e configuracao do indice na nuvem.
from pinecone import Pinecone, ServerlessSpec
# Modelos que convertem TEXTO em VETORES NUMERICOS (embeddings).
from langchain_openai import OpenAIEmbeddings
# Integracao que junta embeddings + gravacao de vetores no Pinecone.
from langchain_pinecone import PineconeVectorStore
# Divisores para quebrar textos longos em partes menores (chunks).
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
# Estrutura LangChain: Document(page_content="texto", metadata={...}).
from langchain_core.documents import Document

# O Hugging Face pode avisar sobre o uso de links simbolicos no Windows.
# Esta configuracao oculta APENAS esse aviso; nao altera o modelo de embeddings.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# ==========================================
# 1. PATH RESOLUTION & SETUP
# ==========================================
# __file__ = caminho deste arquivo Python. Exemplo:
#   C:/meu_projeto/scripts/ingest_sop_pinecone.py
# resolve() obtém o caminho absoluto; parent pega a pasta que contem o script.
script_dir = Path(__file__).resolve().parent

# Como o script esta em /scripts, subir uma pasta leva a raiz do projeto:
#   C:/meu_projeto/scripts -> C:/meu_projeto
# Isso funciona mesmo que o terminal execute o script de outra pasta.
project_root = script_dir.parent

# Le as variaveis definidas em C:/meu_projeto/.env.
# Exemplo de valores possiveis:
#   PINECONE_API_KEY=...
#   Embeddings_model=LOCAL
#   Local_Embedding_Model=BAAI/bge-m3
# A chave nao deve ficar escrita diretamente no codigo nem ser versionada no Git.
load_dotenv(project_root / ".env")

# os.getenv devolve o valor de uma variavel de ambiente ou None, se ausente.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Sem a chave, o script nao consegue criar/acessar o indice remoto.
# "raise" interrompe a execucao imediatamente com uma mensagem explicativa.
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")

# Pasta para guardar o historico local de quais documentos ja foram ingeridos.
# Ex.: C:/meu_projeto/data/cache/
cache_dir = project_root / "data" / "cache"

# Cria a pasta se necessario. parents=True cria tambem pastas intermediarias;
# exist_ok=True nao gera erro quando ela ja existe.
cache_dir.mkdir(parents=True, exist_ok=True)

# Caminho do arquivo JSON que sera o "controle de alteracoes".
# Exemplo: {"seguranca.pdf": "abc123...", "entregas.md": "def456..."}
HASH_CACHE_FILE = cache_dir / "ingestion_hash_cache.json"

# Na primeira execucao, partimos de um dicionario vazio.
# Em execucoes seguintes, recuperamos os hashes salvos anteriormente.
hash_cache = {}
if HASH_CACHE_FILE.exists():
    try:
        # "with" fecha o arquivo automaticamente ao sair do bloco.
        # json.load converte o conteudo JSON em objeto Python (aqui, dicionario).
        with open(HASH_CACHE_FILE, "r") as f:
            hash_cache = json.load(f)
    except Exception:
        # ATENCAO: essa e a regra ORIGINAL. Se o cache estiver ilegivel,
        # ele e considerado vazio, e os arquivos poderao ser reprocessados.
        # Nao confundir esta regra com tratamento de erro que interrompe o script.
        hash_cache = {}

# ==========================================
# 2. DYNAMIC ENVIRONMENT ROUTING
# ==========================================
# Define QUAL modelo criara os vetores:
# - getenv(..., "LOCAL"): usa LOCAL quando nao existe configuracao no .env;
# - strip(): remove espacos antes/depois;
# - upper(): aceita "openai", "OpenAI" e "OPENAI" como o mesmo valor.
EMBEDDINGS_MODEL_SETTING = os.getenv("Embeddings_model", "LOCAL").strip().upper()

# Modelo A: API da OpenAI (embeddings gerados na nuvem).
if EMBEDDINGS_MODEL_SETTING == "OPENAI":
    print("🤖 Mode: Utilizing Cloud OpenAI Embeddings (1536 Dim)...")

    # Cria o objeto que transforma uma string em um vetor numerico.
    # Por exemplo (ILUSTRATIVO): "temperatura ideal" -> [0.12, -0.08, ...].
    # A quantidade real de numeros, na configuracao padrao usada aqui, e 1536.
    # A chamada de rede para embedar textos acontece mais adiante, em add_documents.
    embeddings = OpenAIEmbeddings()

    # Cada modelo deve usar um indice com dimensionalidade compativel.
    # Separar os indices evita misturar vetores de modelos diferentes.
    INDEX_NAME = "fde-sop-index-openai"  # Isolated OpenAI Index
    TARGET_DIMENSION = 1536
else:
    # Modelo B: Hugging Face local (executa na maquina em que o script roda).
    # Se nao houver nome configurado, o codigo usa BAAI/bge-m3.
    # ATENCAO: qualquer valor diferente de "OPENAI" entra neste bloco,
    # inclusive valores digitados incorretamente no .env.
    local_model_target = os.getenv("Local_Embedding_Model", "BAAI/bge-m3").strip()

    print(f"🤗 Mode: Local Fallback Settings Activated. Launching [{local_model_target}] (1024 Dim)...")

    # Importacao condicional: so precisamos desta biblioteca no modo LOCAL.
    from langchain_huggingface import HuggingFaceEmbeddings

    # O objeto carrega o modelo escolhido e depois gerara embeddings de texto.
    # device='cpu' significa executar no processador, sem depender de GPU.
    # OBS.: na primeira vez, o modelo pode precisar ser baixado.
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_target,   # Passes parameter dynamically
        model_kwargs={'device': 'cpu'}
    )

    #Pense no índice como uma coleção onde o Pinecone guarda os vetores dos 
    #documentos. O script usa esse nome para criar ou acessar a coleção e enviar os
    # vetores para ela.

    #O nome inclui local porque esse índice é destinado aos embeddings gerados
    #pelo modelo local. O modo OpenAI usa outro índice. Separá-los evita misturar 
    #vetores de modelos diferentes, que podem ter dimensões incompatíveis.
    INDEX_NAME = "fde-sop-index-local"   # Isolated Local Model Index
    TARGET_DIMENSION = 1024              # Standard width for BGE-M3
    # ATENCAO: 1024 e a dimensao do BGE-M3. Se o .env apontar para OUTRO modelo
    # com tamanho diferente, sera necessario ajustar esta constante.

# ==========================================
# 3. PINECONE PROVISIONING
# ==========================================
# Provisao = garantir que o banco/indice vetorial necessario exista.
print(f"Connecting to Pinecone Index Target: [{INDEX_NAME}]...")

# Autentica o cliente Pinecone utilizando a chave do .env.
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pede ao Pinecone os nomes dos indices existentes na conta.
# Exemplo: ["fde-sop-index-openai", "teste"]
existing_indexes = pc.list_indexes().names()

# Se o indice escolhido ja existe, verificamos se sua dimensao corresponde
# ao vetor do modelo selecionado. Vetor com 1536 numeros nao cabe em indice 1024.
if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != TARGET_DIMENSION:
        print(f"⚠️ Fixing tracking: Purging mismatched {desc.dimension} dim index...")

        # ATENCAO: esta operacao APAGA O INDICE EXISTENTE E TODOS OS SEUS VETORES.
        # Nao e apenas um ajuste de configuracao. Este e o comportamento original.
        pc.delete_index(INDEX_NAME)

        # Atualiza a lista em memoria para que o bloco seguinte recrie o indice.
        existing_indexes = [name for name in existing_indexes if name != INDEX_NAME]

# Se ele nao existe (ou acabou de ser removido), cria o indice.
if INDEX_NAME not in existing_indexes:
    print(f"Creating isolated target index: {INDEX_NAME} ({TARGET_DIMENSION} Dim)...")
    pc.create_index(
        name=INDEX_NAME,               # Identificador do indice no Pinecone.
        dimension=TARGET_DIMENSION,     # Comprimento exigido de cada vetor.
        metric="cosine",               # Compara similaridade pela orientacao dos vetores.
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
        # Hospeda o indice serverless na regiao indicada da AWS.
    )

# index_client: cliente Pinecone "direto", usado mais abaixo para exclusoes.
index_client = pc.Index(INDEX_NAME)

# vector_store: adaptador LangChain que:
#   1) recebe objetos Document;
#   2) extrai o page_content de cada um;
#   3) chama o modelo de embeddings;
#   4) envia os vetores + metadados + IDs para o Pinecone.
# Assim, nao precisamos calcular um vetor e montar um upsert manualmente.
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# ==========================================
# 4. ROBUST POLYMORPHIC PARSER
# ==========================================
# Uma unica funcao consegue ler formatos diferentes (MD, TXT, PDF, CSV, XLSX).
# ENTRADA: caminho do arquivo, por exemplo data/policy/transporte.pdf.
# SAIDA: lista de Documents, um para cada trecho de texto preparado.
def parse_and_chunk_document(doc_path: Path) -> list[Document]:
    # suffix devolve a extensao, como ".PDF"; lower() converte para ".pdf".
    ext = doc_path.suffix.lower()

    # Aqui vamos acumular os trechos extraidos, independentemente do formato.
    raw_chunks: list[Document] = []
    

#chunk_size=600: cada parte terá até cerca de 600 caracteres.
#chunk_overlap=60: partes vizinhas repetem 60 caracteres para não perder o contexto
#na divisão.
#RecursiveCharacterTextSplitter: tenta dividir em pontos naturais, como parágrafos 
#ou espaços, em vez de cortar o texto no meio de uma palavra.
#Por exemplo, se uma parte termina com “temperatura recomendada para”, a 
#seguinte pode começar repetindo esse trecho, mantendo a ideia conectada.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)

    if ext == ".md":
        # Em Markdown, "#" / "##" / "###" representam niveis de titulo.
        # Exemplo:
        #   # Transporte            -> Header_1 = "Transporte"
        #   ## Temperatura          -> Header_2 = "Temperatura"
        #   ### Produtos congelados -> Header_3 = "Produtos congelados"
        # Esses titulos viram METADADOS e ajudam a identificar cada secao.
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]

        # Ele usa a lista headers_to_split_on — configurada logo acima — para separar
        # o texto quando encontra títulos como #, ## ou ###. Os títulos encontrados também são 
        # guardados como metadados de cada parte, ajudando a identificar o assunto.

        # Por exemplo, um arquivo com ## Armazenamento e ## Transporte será dividido 
        # em seções com esses títulos associados. Depois, o text_splitter pode dividir 
        # essas seções em partes menores, se necessário.

        #Essa linha não divide o texto ainda. Ela cria e configura o separador Markdown,
        #  guardando-o na variável md_splitter

        #headers_to_split_on é a lista definida logo acima, que diz quais 
        # títulos reconhecer: #, ## e ###. Mais adiante, a chamada

        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Le todo o texto do .md e primeiro separa por estrutura de titulos.
        raw_text = doc_path.read_text(encoding="utf-8")

        # separa esse texto em seções pelos títulos (#, ##, ###). Cada seção vira 
        # um documento, e o título correspondente fica registrado nos metadados.
        header_docs = md_splitter.split_text(raw_text)

        # Depois quebra secoes muito grandes em blocos menores.
        # split_documents preserva os metadados criados pelo md_splitter.

        #pega essas seções e divide as que forem longas demais em partes de até 
        #aproximadamente 600 caracteres.

        #overlap: Ele repete cerca de 60 caracteres do final de uma parte no começo 
        #da próxima, para preservar o contexto entre elas.

        #ex: Por exemplo, se uma seção tiver 1.000 caracteres, ela será dividida em
        # blocos; o segundo bloco começa repetindo um trecho do primeiro. Se a seção
        # tiver menos de 600 caracteres, normalmente fica em um único bloco.
        raw_chunks = text_splitter.split_documents(header_docs)

    elif ext == ".txt":
        # TXT nao tem metadados de cabecalho; comecamos com um Document unico.
        raw_text = doc_path.read_text(encoding="utf-8")
        raw_docs = [Document(page_content=raw_text)]

        # Se o arquivo tem 2500 caracteres, ele vira varios Documents pequenos.
        raw_chunks = text_splitter.split_documents(raw_docs)

    elif ext == ".pdf":
        # Cada pagina vira um Document, com a pagina original nos metadados.
        pdf_docs = []
        try:
            # "rb" = abrir em modo binario, necessario para o leitor de PDF.
            with open(doc_path, "rb") as f:
                reader = pypdf.PdfReader(f)

                # enumerate(...): percorre as paginas e devolve 0, 1, 2, ...
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()

                    # Descarta paginas das quais nenhum texto util foi extraido.
                    if page_text and page_text.strip():
                        # +1 faz a numeracao comecar na pagina 1 para o usuario.
                        pdf_docs.append(Document(page_content=page_text, metadata={"page_number": page_num + 1}))

            # Divide o conteudo de cada pagina em trechos menores; o numero
            # da pagina continua associado a cada pedaco.
            raw_chunks = text_splitter.split_documents(pdf_docs)
        except Exception as e:
            print(f"  ❌ Error parsing PDF {doc_path.name}: {e}")
            return []  # Em erro de leitura, nao devolve documentos para indexar.

        # LIMITACAO: extract_text nao faz OCR. PDF escaneado como imagem pode
        # devolver texto vazio, mesmo contendo palavras visiveis na imagem.

    elif ext in [".csv", ".xlsx"]:
        # DataFrame = estrutura de tabela do pandas (linhas e colunas).
        try:
            # Expressao condicional: se for CSV, read_csv; senao, read_excel.
            df = pd.read_csv(doc_path) if ext == ".csv" else pd.read_excel(doc_path)
        except Exception as e:
            print(f"  ❌ Error reading table: {e}")
            return []

        # Cada LINHA da planilha vira UM Document (aqui nao se usa text_splitter).
        # Exemplo: colunas [produto, temperatura_min, temperatura_max]
        # linha [vacina, 2, 8] -> "produto: vacina | temperatura_min: 2 | ..."

        #idx recebe o índice da linha. 
        #row recebe os valores daquela linha, junto com os nomes das colunas.
        #produto | temperatura |
        #Vacina  | 2           | 
        for idx, row in df.iterrows():
            # row (Series) vira dicionario {nome_da_coluna: valor_da_celula}.
            row_dict = row.to_dict()

            # List comprehension: percorre cada coluna da linha e produz
            # "nome_coluna: valor", descartando valores nulos e strings vazias.
            row_items = [
                f"{str(col)}: {str(val)}"
                for col, val in row_dict.items()
                if pd.notna(val) and str(val).strip() != ""
            ]

            if row_items:
                # Junta os campos em uma unica frase para embeddar a linha inteira.
                row_text = " | ".join(row_items)

                # row_index guarda o indice do DataFrame para rastrear a origem.
                # OBS.: por padrao, a primeira linha de dados tem indice 0.
                doc_item = Document(page_content=row_text, metadata={"row_index": int(idx)})
                
                #Aqui, doc_item representa uma linha da tabela convertida em documento. Essa linha a 
                #adiciona à lista para que, depois, o script possa processar e enviar todos os documentos ao Pinecone.
                #É como uma caixa: cada vez que o código executa essa linha, coloca mais um documento dentro dela.


                raw_chunks.append(doc_item)

    # Ultima limpeza, comum a todos os formatos:
    # - strip remove espacos/quebras de linha nas extremidades;
    # - se ficar vazio, nao enviamos aquele trecho ao modelo nem ao Pinecone.
    valid_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.strip()
        if clean_text:
            chunk.page_content = clean_text
            valid_chunks.append(chunk)

    # Exemplo de retorno:
    # [Document(page_content="Armazenar entre 2 e 8 C", metadata={"page_number": 3}), ...]
    return valid_chunks

# ==========================================
# 5. INCREMENTAL PIPELINE WITH BATCHING
# ==========================================
# Incremental = processar arquivos NOVOS ou ALTERADOS, evitando retrabalho.
# Pasta de origem, onde os SOPs e politicas devem estar guardados.
policy_dir = project_root / "data" / "policy"

# glob procura nomes que se encaixam em cada padrao. "*" = qualquer nome.
# Exemplo "*.pdf" encontra "procedimento.pdf", mas nao "foto.png".
target_patterns = ["*.md", "*.txt", "*.pdf", "*.csv", "*.xlsx"]

# Dicionario {nome_do_arquivo: caminho_completo_do_arquivo}.
current_files = {}
for pattern in target_patterns:
    for file_path in policy_dir.glob(pattern):
        current_files[file_path.name] = file_path
# OBS.: glob("*.pdf") busca na pasta indicada; nao varre subpastas recursivamente.

print(f"Found {len(current_files)} policy file(s) in {policy_dir}...")

# Registro que sera salvo como cache ao final desta execucao.
updated_cache = {}

# Flag booleana: indica se o script detectou algo que exige gravar o JSON.
cache_modified = False

# Convertemos nomes em sets porque set permite calcular diferenca entre grupos.
# Exemplo: cache antigo = {"A.pdf", "B.pdf"}
#          arquivos atuais = {"A.pdf", "C.pdf"}
#          deleted_files = {"B.pdf"}
cached_filenames = set(hash_cache.keys())
current_filenames = set(current_files.keys())
deleted_files = cached_filenames - current_filenames

# Se um arquivo foi REMOVIDO da pasta, tentamos apagar seus vetores do Pinecone.
for deleted_file in deleted_files:
    print(f"🗑️ Detected deleted file: {deleted_file}. Purging from Pinecone...")
    try:
        # Filtro por METADADOS: remove vetores que guardam source_file=B.pdf.
        # Nao esta apagando o arquivo local; ele ja nao existe na pasta.
        index_client.delete(filter={"source_file": {"$eq": deleted_file}})
        cache_modified = True
    except Exception as e:
        print(f"  ❌ Failed to purge {deleted_file}: {e}")

# Agora percorremos cada documento que esta presente na pasta.
for file_name, file_path in current_files.items():
    # read_bytes le os bytes do arquivo, inclusive formato e pequenas alteracoes.
    file_bytes = file_path.read_bytes()

    # MD5 gera uma assinatura fixa do conteudo. Se mudar 1 byte,
    # normalmente a assinatura muda, permitindo detectar uma nova versao.
    # MD5 e usado aqui para detectar ALTERACOES, nao para seguranca criptografica.
    #hexadecimais, como "5d41402abc4b2a76b9719d911017c592".

    #O script compara essa assinatura com a salva no cache. Se for igual, 
    #entende que o conteúdo não mudou e pode pular o reprocessamento; 
    #se for diferente, processa o arquivo novamente.
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Anota o hash atual, mesmo antes de testar se houve alguma mudanca.
    updated_cache[file_name] = file_hash

    # Se o hash antigo e igual ao atual, nao precisa gerar embeddings novamente.
    # Exemplo: o mesmo "B.pdf" rodou ontem e hoje sem nenhuma mudanca.
    if hash_cache.get(file_name) == file_hash:
        print(f"✨ Skipped (Unchanged): {file_name}")
        continue  # Pula o restante do loop deste arquivo; vai para o proximo.

    # Arquivo novo (nao havia hash) ou alterado (hash diferente).
    print(f"🔄 Processing updates: {file_name}...")
    cache_modified = True

    try:
        try:
            # Remove os chunks da versao anterior para evitar trechos obsoletos.
            # Ex.: se antes existiam 10 chunks e agora existem 7,
            # precisamos remover os 3 antigos que sobrariam.
            index_client.delete(filter={"source_file": {"$eq": file_name}})
        except Exception:
            # Comportamento original: ignora qualquer falha nesta exclusao.
            # ATENCAO: isso pode deixar vetores antigos no indice.
            pass

        # Chama o parser da secao 4 e devolve uma lista de Document.

        #identifica o formato do arquivo (PDF, Markdown, CSV etc.), extrai o 
        #conteúdo e o divide em partes menores. O resultado — uma lista de 
        #documentos — fica guardado em chunks.
        chunks = parse_and_chunk_document(file_path)

        # Nada para embeddar (por exemplo, PDF escaneado sem camada de texto).
        if not chunks:
            print(f"  ⚠️ No valid text chunks extracted from {file_name}.")
            continue

        # IDs manuais e metadata: ajudam a identificar e filtrar cada chunk.
        explicit_ids = []
        for idx, chunk in enumerate(chunks):
            # Metadados seguem junto com o vetor no banco vetorial.
            # Ex.: source_file = "sop_transporte.pdf".
            chunk.metadata["source_file"] = file_name
            chunk.metadata["file_format"] = file_path.suffix.replace(".", "").upper()
            chunk.metadata["document_type"] = "Compliance Asset"

            # ID deterministico: "sop_transporte.pdf-chunk-0", "...-chunk-1"...
            # Assim, o identificador da mesma posicao nao muda entre execucoes.
            explicit_ids.append(f"{file_name}-chunk-{idx}")

        # Envio em lotes para nao tentar subir milhares de chunks de uma vez.
        batch_size = 100
        total_chunks = len(chunks)
        print(f"  📤 Upserting {total_chunks} chunk(s) in batches of {batch_size}...")

        # range(0, total_chunks, 100) devolve 0, 100, 200, ...
        # Exemplo: 235 chunks -> lotes [0:100], [100:200], [200:300].
        for i in range(0, total_chunks, batch_size):
            # Fatiamento [inicio:fim]: fim nao e incluido.

            #chunks contém os documentos divididos.
            #explicit_ids contém um ID correspondente a cada documento.
            #i marca o início do lote e batch_size define quantos itens pegar.

            #i : i + batch_size é uma fatia da lista: começa em i e vai até antes 
            #do índice final.

            #Por exemplo, se i = 0 e batch_size = 100, pega os itens de índice 0 a 
            #99. Na próxima volta, i avança para 100 e pega os próximos.
            
            #  As duas 
            #linhas usam os mesmos índices para manter cada documento junto do seu 
            # ID correspondente.
            batch_docs = chunks[i : i + batch_size]
            batch_ids = explicit_ids[i : i + batch_size]

            # PONTO-CHAVE DO RAG: LangChain gera os embeddings a partir do
            # page_content de cada Document e faz upsert no Pinecone.
            # "upsert" = inserir vetor novo OU atualizar vetor com mesmo ID.
            # Metadados (pagina, arquivo, secao etc.) seguem com o vetor.
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    except Exception as e:
        # Se alguma etapa de um arquivo falhar, loga e passa para os demais.
        print(f"❌ Error during ingestion of {file_name}: {e}")

        # Retira o hash da lista nova para tentar reingeri-lo na proxima vez.
        updated_cache.pop(file_name, None)
        # OBS.: lotes ja enviados antes da falha podem continuar no Pinecone.

# ==========================================
# 6. SYNC HASH CACHE
# ==========================================
# Somente quando ocorreu uma mudanca, regravamos o controle de hashes.
if cache_modified:
    with open(HASH_CACHE_FILE, "w") as f:
        # indent=4 torna o JSON legivel (quebras de linha + recuo).
        json.dump(updated_cache, f, indent=4)
    print("✅ Ingestion & cache update complete.")
else:
    # Se todos os arquivos mantiveram o mesmo conteudo, nao gera embeddings.
    print("🌴 Index is already up-to-date.")

# OBSERVACOES SOBRE O COMPORTAMENTO ORIGINAL (SEM ALTERAR A LOGICA):
# 1) Se o parser devolver [] depois de uma exclusao, o hash desse arquivo
#    pode ser salvo mesmo sem vetores; uma proxima execucao podera ignora-lo.
# 2) Se a remocao de arquivo do Pinecone falhar, o cache atualizado pode
#    perder a referencia ao arquivo, e a remocao talvez nao seja tentada de novo.
# 3) O script e um processo de INGESTAO (offline), nao um chat RAG completo.
#    A busca por similaridade e a geracao da resposta ficam em outra etapa.
