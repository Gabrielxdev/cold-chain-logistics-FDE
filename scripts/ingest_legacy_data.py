# Objetivo: ler um CSV, adaptar suas colunas ao formato de um sistema legado
# (um sistema antigo, com nomes abreviados) e gravar o resultado no SQL Server.
# Fluxo: CSV -> tabela em memória (DataFrame) -> adaptação -> tabela SQL.
# As instruções abaixo são executadas em sequência ao executar este arquivo.

# Path permite trabalhar com caminhos de arquivos e pastas.
from pathlib import Path
# pandas manipula tabelas; "pd" é o apelido usado para acessar suas funções.
import pandas as pd
# urllib oferece utilitários de URL; aqui usamos parse para codificar a conexão.
import urllib
# os permite ler variáveis de ambiente do processo.
import os
# create_engine cria o objeto que gerencia o acesso ao banco via SQLAlchemy.
from sqlalchemy import create_engine
# load_dotenv carrega configurações de um arquivo .env para o ambiente.
from dotenv import load_dotenv

# 1. Localizar os arquivos e carregar as configurações.
# __file__ é o caminho deste script; Path transforma esse caminho em um objeto.
# resolve() obtém o caminho absoluto e parent pega a pasta que contém o arquivo.
# Neste projeto, o resultado é a pasta scripts/.
script_dir = Path(__file__).resolve().parent
# parents[0] é a primeira pasta acima de scripts/: a raiz do projeto.
# Isso evita depender da pasta em que o terminal estava ao iniciar o script.
project_root = script_dir.parents[0]

# O operador / de Path junta partes de um caminho; aqui aponta para o .env da raiz.
# Por padrão, load_dotenv não sobrescreve variáveis já existentes no ambiente.
load_dotenv(project_root / ".env")

# Monta o caminho do CSV dentro de data/raw/; ainda não abre nem lê o arquivo.
data_path = project_root / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"

# getenv procura a variável pelo nome; o segundo argumento é o valor padrão.
# Se SQL_SERVER_HOST não existir, usa localhost (o ambiente onde o script roda).
db_host = os.getenv("SQL_SERVER_HOST", "localhost")
# Define a porta de conexão; "1433" é o padrão aqui e é armazenado como texto.
db_port = os.getenv("SQL_SERVER_PORT", "1433")
# Sem segundo argumento, getenv retorna None caso a variável não exista.
# O usuário deve estar configurado no ambiente ou no .env carregado acima.
db_user = os.getenv("SQL_ADMIN_USER")
# Lê a senha do ambiente para não escrevê-la diretamente no código-fonte.
db_password = os.getenv("SQL_ADMIN_PASSWORD")

# 2. Ler os dados de origem.
# print exibe o andamento; o prefixo f insere o valor da variável entre as chaves.
print(f"Loading CSV from {data_path}...")
# read_csv lê o arquivo e cria um DataFrame: uma tabela em memória com linhas e colunas.
# df guarda essa tabela. O pandas infere os tipos dos dados durante a leitura.
df = pd.read_csv(data_path)

# 3. Adaptar as colunas para o formato esperado pelo sistema legado.
# Um dicionário usa pares chave: valor. Aqui: nome original no CSV: novo nome no banco.
# O mapeamento troca apenas os nomes; não converte unidades nem valida os valores.
legacy_mapping = {
    # Data/hora -> TS_UTC. O nome sugere UTC, mas não converte o fuso horário.
    'timestamp': 'TS_UTC',
    # Latitude do GPS do veículo -> abreviação de latitude.
    'vehicle_gps_latitude': 'V_LAT',
    # Longitude do GPS do veículo -> abreviação de longitude.
    'vehicle_gps_longitude': 'V_LON',
    # Temperatura do sensor IoT -> nome que indica temperatura em graus Celsius.
    'iot_temperature': 'IOT_TEMP_VAL_C',
    # Estado da carga -> coluna de condição da carga, mantendo os valores originais.
    'cargo_condition_status': 'CGO_COND_CD',
    # Classificação de risco -> coluna de classificação textual de risco.
    'risk_classification': 'RISK_CLS_TXT',
    # Probabilidade de atraso -> nome que sugere um valor decimal.
    'delay_probability': 'DELAY_PROB_DEC',
    # Nível de congestionamento do porto -> nome abreviado na tabela de destino.
    'port_congestion_level': 'PRT_CNG_LVL',
    # Nível de risco da rota -> coluna de índice de risco da rota.
    'route_risk_level': 'RT_RSK_IDX'
}

# keys() fornece os nomes originais; list(...) transforma esses nomes em uma lista.
# df[lista] seleciona somente essas colunas, na ordem do dicionário, e todas as linhas.
# Se uma coluna listada não existir no CSV, essa seleção gera um erro (KeyError).
# rename(columns=...) aplica as trocas de nomes e retorna a tabela adaptada.
# df mantém os nomes originais; df_legacy recebe o resultado da transformação.
df_legacy = df[list(legacy_mapping.keys())].rename(columns=legacy_mapping)

# Cria uma coluna e preenche TODAS as linhas com 'Y' (de "yes", ou "sim").
# É uma marca simulada de ingestão; ainda não confirma que houve gravação no banco.
df_legacy['SYS_INGEST_FLAG'] = 'Y'

# 4. Preparar o acesso ao SQL Server (por exemplo, a instância em Docker).
# Esta mensagem informa a etapa; ainda não confirma uma conexão bem-sucedida.
print("Connecting to legacy MSSQL Database...")


#monta uma string de conexão: um texto com as informações que o programa
#precisa para se conectar ao SQL Server.
connection_string = (
        # {{ e }} numa f-string produzem chaves literais ao redor do nome do driver.
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        # No formato ODBC do SQL Server, a vírgula separa endereço e porta.
        f"SERVER={db_host},{db_port};"
        # Escolhe master como banco de destino; não cria um banco separado.
        f"DATABASE=master;"
        # UID informa o usuário para autenticação no SQL Server.
        f"UID={db_user};"
        # PWD informa a senha; o texto montado contém essa credencial.
        f"PWD={db_password};"
        # Configura o cliente para não exigir criptografia da conexão.
        f"Encrypt=no;"
        # Permite confiar no certificado do servidor sem validá-lo quando aplicável.
        f"TrustServerCertificate=yes;"
    )

# Codifica espaços e caracteres especiais para transportar a string dentro de uma URL.
# Codificação de URL não é criptografia: a senha continua recuperável desse texto.

# Esta linha codifica a string de conexão para que ela possa ser colocada dentro de uma URL.
# Caracteres como espaços, ; e = têm significados especiais em URLs, então precisam ser representados de forma segura.
# Por exemplo, espaços viram + e ; vira %3B.
params = urllib.parse.quote_plus(connection_string)

#Essa linha cria um engine do SQLAlchemy, que é o objeto usado para gerenciar conexões com o banco:

#mssql indica que o banco é SQL Server.
#pyodbc indica o driver Python que será usado.
#odbc_connect={params} passa a string ODBC codificada.
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# 5. Gravar os dados transformados na tabela de destino.
# Guarda o nome da tabela para reutilizá-lo na mensagem e na gravação.
table_name = 'TBL_SC_FLEET_HIST_RAW'
# Mostra qual tabela receberá os dados antes de iniciar a operação.
print(f"Ingesting into {table_name}. This may take a minute...")

# to_sql envia as linhas e colunas de df_legacy ao banco usando engine.
# table_name: nome da tabela que receberá os dados.
# if_exists='replace': se a tabela existir, ela é removida e recriada com estes dados.
# Portanto, executar novamente SUBSTITUI a tabela anterior; não acrescenta ao histórico.
# index=False: não grava o índice do DataFrame (rótulos das linhas) como coluna extra.
# schema='dbo': usa o esquema dbo, que agrupa objetos dentro do banco master.
# Sem dtype explícito, pandas/SQLAlchemy inferem os tipos das colunas SQL.
df_legacy.to_sql(table_name, engine, if_exists='replace', index=False, schema='dbo')

# Esta mensagem só aparece se to_sql terminar sem erro e a execução chegar até aqui.
print("✅ Legacy data ingestion complete!")
