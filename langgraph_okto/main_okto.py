import shutil
import uuid
from dotenv import load_dotenv 
from tools_okto import update_info_user,_print_event
from agent_okto import ApostasOnlineAgent
from langchain.agents import AgentExecutor
from agent_okto import part_1_graph

load_dotenv()

def main():
    agente = ApostasOnlineAgent()
    executor = AgentExecutor(agent=agente.agente,
                            tools=agente.tools,
                            verbose=True)
    
    while True:
        tutorial_questions = [
        "Quais são as melhores estratégias de apostas para o próximo jogo de futebol?",
        "Como posso acompanhar as odds ao vivo durante uma partida?",
        "Qual é o limite máximo de aposta para a próxima corrida de cavalos?",
        "Quais são os bônus disponíveis para novos apostadores?",
        "Como posso fazer uma aposta múltipla em diferentes esportes?",
        "Qual é a política de reembolso para apostas não concluídas?",
        "Há algum limite para apostas em jogos de cassino online?",
        "Como posso verificar o status das minhas apostas atuais?",
        ]
        
        if tutorial_questions.lower() == 's':
            print("Encerrando o chat...")
            break

        resposta = executor.invoke({"input": tutorial_questions})
        print(resposta)
        # Update with the backup file so we can restart from the original place in each section
        db = update_info_user(db)
        thread_id = str(uuid.uuid4())

        config = {
            "configurable": {
                # The passenger_id is used in our flight tools to
                # fetch the user's flight information
                "passenger_id": "3442 587242",
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id,
            }
        }


        _printed = set()
        for question in tutorial_questions:
            events = part_1_graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)


        _printed = set()
        for question in tutorial_questions:
            events = part_1_graph.stream(
                {"messages": ("user", question)}, config, stream_mode="values"
            )
            for event in events:
                _print_event(event, _printed)
if __name__ == "__main__":
    main()