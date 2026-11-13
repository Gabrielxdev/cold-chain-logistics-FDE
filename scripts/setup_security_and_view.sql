-- ====================================================================
-- Script de segurança e camada semântica do projeto FDE.
-- Objetivo: expor ao agente uma interface de leitura mais clara e limitar
-- as permissões da conta usada por ele no banco de dados.
-- ====================================================================

-- 1. Cria um schema (um agrupador lógico de objetos) para as views do agente.
CREATE SCHEMA FDE_VIEWS;
-- GO encerra este lote de comandos; é um separador entendido por ferramentas SQL Server.
GO

-- 2. Cria uma view semântica: uma camada de leitura com nomes mais compreensíveis.
CREATE VIEW FDE_VIEWS.VW_ACTIVE_FLEET AS
SELECT 
    TS_UTC AS [Timestamp], -- Renomeia o instante da medição para "Timestamp".
    V_LAT AS [Latitude], -- Expõe a latitude com um nome claro.
    V_LON AS [Longitude], -- Expõe a longitude com um nome claro.
    CAST(IOT_TEMP_VAL_C AS FLOAT) AS [Current_Temperature_C], -- Converte a temperatura para FLOAT e dá um nome descritivo.
    CGO_COND_CD AS [Cargo_Condition_Code], -- Expõe o código da condição da carga.
    RISK_CLS_TXT AS [Risk_Classification], -- Expõe a classificação de risco.
    DELAY_PROB_DEC AS [Delay_Probability], -- Expõe a probabilidade de atraso.
    PRT_CNG_LVL AS [Port_Congestion_Level], -- Expõe o nível de congestionamento do porto.
    RT_RSK_IDX AS [Route_Risk_Index] -- Expõe o índice de risco da rota.
FROM dbo.TBL_SC_FLEET_HIST_RAW; -- Lê os dados da tabela original no schema dbo.
-- Encerra o lote que criou a view.
GO

-- 3. Cria a credencial de servidor (login) usada para autenticação no SQL Server.
-- A senha literal abaixo é apenas um exemplo: use um segredo forte e gerenciado fora do código.
CREATE LOGIN USR_FDE_RO WITH PASSWORD = 'Agente123@';
-- Cria o usuário dentro do banco atual e associa-o ao login do servidor.
CREATE USER USR_FDE_RO FOR LOGIN USR_FDE_RO;
-- Encerra o lote de criação da conta.
GO

-- 4. Permite consultar a view semântica.
GRANT SELECT ON FDE_VIEWS.VW_ACTIVE_FLEET TO USR_FDE_RO;
-- Nega explicitamente a leitura direta da tabela original.
DENY SELECT ON dbo.TBL_SC_FLEET_HIST_RAW TO USR_FDE_RO;
-- Nega operações de escrita e alteração em todos os objetos do schema dbo.
-- Isso não nega SELECT em todo o schema; a leitura da tabela original é negada acima.
DENY INSERT, UPDATE, DELETE, ALTER ON SCHEMA::dbo TO USR_FDE_RO;
-- Encerra o lote de permissões.
GO

-- Sobre SCHEMA::dbo:
-- dbo é um schema (agrupador lógico padrão), não uma pasta física.
-- Esta negação vale para INSERT, UPDATE, DELETE e ALTER nos objetos atuais e futuros
-- desse schema. Ela não impede automaticamente toda leitura; as regras de SELECT
-- deste script são aplicadas especificamente à view e à tabela de origem.