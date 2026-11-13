"""Demonstra o cálculo da busca vetorial com números fictícios.

Não carrega modelos, não calcula embeddings reais e não acessa serviços.
Execute a partir da raiz: python scripts/demo_similaridade_vetorial.py
"""

from math import sqrt


def cosseno(a, b):
    """Compara a direção de dois vetores não nulos de mesma dimensão."""
    if len(a) != len(b):
        raise ValueError("Os vetores precisam ter a mesma dimensão.")
    norma_a = sqrt(sum(x * x for x in a))
    norma_b = sqrt(sum(y * y for y in b))
    if norma_a == 0 or norma_b == 0:
        raise ValueError("A similaridade de cosseno exige vetores não nulos.")
    return sum(x * y for x, y in zip(a, b)) / (norma_a * norma_b)


def main():
    pergunta = "A carga de perecíveis está a 6 graus. Qual procedimento seguir?"
    # TODOS os vetores abaixo foram escolhidos manualmente para ensinar a conta.
    # Eles não são saídas do BGE-M3 e não demonstram a qualidade desse modelo.
    vetor_pergunta = [1.0, 0.2, 0.0]
    trechos = [
        {
            "id": "temperatura",
            "vetor": [0.9, 0.1, 0.0],
            "texto": (
                "Perecíveis: manter de 0 a 4 °C. Acima de 4 °C, contatar o "
                "motorista para reiniciar a unidade auxiliar de refrigeração. "
                "Se o atraso de ETA superar 1 hora, desviar para a instalação "
                "de armazenamento refrigerado de emergência mais próxima."
            ),
        },
        {
            "id": "congestionamento",
            "vetor": [0.1, 0.9, 0.0],
            "texto": (
                "Congestionamento do porto acima de 7: desviar as remessas "
                "ativas para o Inland Empire Overflow Depot."
            ),
        },
        {
            "id": "risco",
            "vetor": [0.0, 0.1, 1.0],
            "texto": (
                "High Risk com probabilidade de atraso acima de 0.65: "
                "escalar para o Tier 2 Logistics Manager."
            ),
        },
    ]

    ranking = sorted(
        ((cosseno(vetor_pergunta, trecho["vetor"]), trecho) for trecho in trechos),
        key=lambda resultado: resultado[0],
        reverse=True,
    )

    print("DEMONSTRAÇÃO: vetores fictícios; textos resumidos do SOP do projeto.")
    print(f"Pergunta: {pergunta}\n")
    for posicao, (score, trecho) in enumerate(ranking, start=1):
        print(f"{posicao}. {trecho['id']}: cosseno = {score:.4f}")

    top_k = 1
    print(f"\nTextos recuperados com top_k={top_k}:")
    for score, trecho in ranking[:top_k]:
        print(f"[{trecho['id']}] {trecho['texto']}")
    print("\nA recuperação termina aqui. Nenhum LLM foi chamado para gerar resposta.")


if __name__ == "__main__":
    main()
