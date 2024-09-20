from dotenv import load_dotenv 
from agent_okto_2 import AgenteOpenAIFunctions
from langchain.agents import AgentExecutor

load_dotenv()

def main():
    agente = AgenteOpenAIFunctions()
    executor = AgentExecutor(agent=agente.agente,
                            tools=agente.tools,
                            verbose=True)
    
    while True:
        pergunta = input("Digite o que está procurando ou digite 's' para sair: ")
        
        if pergunta.lower() == 's':
            print("Encerrando o chat...")
            break

        resposta = executor.invoke({"input": pergunta})

        if "Informação não encontrada" in resposta['output']:
            print("Parece que a informação não foi encontrada. Por favor, tente reformular sua pergunta.")
        else:
            print(resposta['output'])

if __name__ == "__main__":
    main()