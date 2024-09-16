from dotenv import load_dotenv 
from agent_betano import AgenteOpenAIFunctions
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
        print(resposta)
        
if __name__ == "__main__":
    main()