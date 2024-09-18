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

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
class AgenteOpenAIFunctions:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4",
                              temperature=0.5,
                              api_key=os.getenv("OPENAI_API_KEY"))

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
                docs=self.txt_policies_reader._docs,
                func=self.txt_policies_reader.query,
                description="""Consulta as polítics da companhia para verificar se as coisas pedidas ou citadas pelo usuário, podem ser executadas ou não.
                    Use isso antes de escrever qualquer evento para o usuário, se excessão.""",
                return_direct=False
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful customer support assistant for Betano. "
                    " Use the provided tools to search for user_id, company policies, and other information to assist the user's queries. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up."
                    "\n\nCurrent user:\n\n{user_info}\n"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())
        
        self.agente = prompt | self.llm.bind_tools(self.tools)

        builder = StateGraph(State)
        builder.add_node("assistant", self._assistant_node())
        builder.add_node("tools", self._tools_node())
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        memory = MemorySaver()
        self.part_1_graph = builder.compile(checkpointer=memory)

    def _assistant_node(self):
        return Assistant(part_1_assistant_runnable)

    def _tools_node(self):
        return create_tool_node_with_fallback(part_1_tools)

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
