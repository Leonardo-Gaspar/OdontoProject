from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from reader_txt import TxtReader
from datetime import datetime
from langchain.agents import Tool
from tools_betano import VectorStoreRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, Assistant, create_tool_node_with_fallback, part_1_assistant_runnable, part_1_tools
import os
import json
import uuid
from tools_betano import update_info_user, _print_event
from langchain_chains import RetrievalQA
from IPython.display import Image, display

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Definição do gráfico de estado
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Ajuste para perguntas relacionadas a apostas online
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

# Configuração do gráfico
class ApostasOnlineAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=OPENAI_API_KEY)
        self.txt_documents_reader = TxtReader()
        self.txt_policies_reader = VectorStoreRetriever()

        self.tools = [
            Tool(
                name=self.txt_documents_reader.name,
                func=self.txt_documents_reader._run,
                description=self.txt_documents_reader.description,
                return_direct=False
            ),
            Tool(
                name="txt_policies_reader",
                func=self.txt_policies_reader.query,
                description="Busca informações sobre apostas e políticas da empresa.",
                return_direct=False
            )
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Você é um assistente especializado em apostas online. Use as ferramentas fornecidas para buscar informações sobre apostas, bônus, políticas da empresa e outros detalhes relacionados. Se a primeira busca não retornar resultados, expanda o escopo da busca."
                    "\n\nUsuário atual:\n\n{user_info}\n"
                    "\nHora atual: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

        self.agente = prompt | self.llm.bind_tools(self.tools)

        # Definição do StateGraph
        builder = StateGraph(State)
        builder.add_node("assistant", Assistant(part_1_assistant_runnable))
        builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        # Configuração da memória para o gráfico
        memory = MemorySaver()
        self.part_1_graph = builder.compile(checkpointer=memory)

        # Exibir o gráfico
        self._display_graph()

    def _display_graph(self):
        try:
            display(Image(self.part_1_graph.get_graph(xray=True).draw_mermaid_png()))
        except Exception:
            # Isso requer algumas dependências extras e é opcional
            pass

    def _run(self, input_text: str) -> str:
        resposta = self.agente.invoke({"input": input_text})
        return json.dumps(resposta)

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_info": user_id}
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

    @staticmethod
    def criar_qa_chain(documentos):
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos = quebrador.split_documents(documentos)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(textos, embeddings)

        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

        return qa_chain

    @staticmethod
    def responder_pergunta(qa_chain, pergunta):
        resultado = qa_chain.invoke({'query': pergunta})
        return resultado

# Atualização do banco de dados e configuração
db = update_info_user(db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "user_id": "12345",
        "thread_id": thread_id,
    }
}

_printed = set()
agent = ApostasOnlineAgent()  # Instancia o agente

for question in tutorial_questions:
    events = agent.part_1_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
