from dotenv import load_dotenv
from reader_txt import carregar_documentos
from agent_betano import criar_agente, responder_pergunta

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

def main():
    documentos = carregar_documentos()

    agente = criar_agente(documentos)

    pergunta = input("Qual pergunta deseja fazer? ")
    
    # Obtém a resposta
    resposta = responder_pergunta(agente, pergunta)
    print(resposta)

if __name__ == "__main__":
    main()