import sqlite3
from langchain_core.tools import tool 
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
import re 
import requests
import numpy as np
import openai
from langchain_core.tools import tool

#Esta faltando o link do api 
response = requests.get("https://link-da-api.com")
response.raise_for_status()
faq_text = response.text

#Aqui ficaria os documentos em txt das regras da betano
policies_documents = [{"policies_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

class VectorStoreRetriever:
    def __init__(self, policies_documents: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = policies_documents
        self._client = oai_client

    @classmethod
    def from_docs(cls, policies_documents, oai_client):
        embeddings = oai_client.embeddings.create(model="text-embedding-3-small", 
                                                input=[policy_document["policies_content"] for policy_document in policies_documents]
                                                )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(policies_documents, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]

retriever = VectorStoreRetriever.from_docs(policies_documents, openai)
@tool
def lookup_policy(query: str) -> str:
    """Consulta as polítics da companhia para verificar se as coisas pedidas ou citadas pelo usuário,
    podem ou não.
    Use isso antes de escrever qualquer evento para o usuário, se excessão."""
    policies_documents = retriever.query(query, k=2)
    return "\n\n".join([policy_document["policies_content"] for policy_document in policies_documents])

@tool
def add_user(user_info: str, *, config: RunnableConfig) -> str:
    """Adiciona um novo usuário ao banco de dados com um ID gerado automaticamente e as informações fornecidas.
        Returns:
            Uma lista de dicionários cada um contendo detalhes do usuário.
    """

    db = "caminho_para_banco_de_dados.db" #Faltando caminho para DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(user_id) FROM user_info")
    max_id = cursor.fetchone()[0]
    new_id = (max_id or 0) + 1
    
    cursor.execute(
        "INSERT INTO user_info (user_id, user_info) VALUES (?, ?)",
        (new_id, user_info)
    )
    
    conn.commit()
    
    configuration = config.get("configurable", {})
    if "user_id" in configuration:
        configuration["user_id"] = new_id
        config.set("configurable", configuration)
    
    conn.close()
    
    return f"Usuário com ID {new_id} adicionado com sucesso."


@tool
def fetch_user_betano_information(user_id: str, config: RunnableConfig) -> list[dict]:
    """Busca as informações do usuário e histórico do usuário, no banco de dados, com base no user_id fornecido.
    Returns:
        Uma lista de dicionários cada um contendo detalhes do usuário,
        associados ao seu histórico and bet details. 
    """
    
    configuration = config.get("configurable", {})
    configured_user_id = configuration.get("user_id", None)
    
    if configured_user_id is None:
        raise ValueError("Nenhum user_id configurado.")
    
    if user_id != configured_user_id:
        raise ValueError(f"O ID do usuário fornecido ({user_id}) não corresponde ao ID configurado ({configured_user_id}).")
    
    db = "caminho_para_banco_de_dados.db" #Faltando o caminho da DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT user_info FROM user_info WHERE user_id = ?", (user_id,))
    user_info = cursor.fetchone()
    
    conn.close()
    
    if user_info is None:
        return f"Usuário com ID {user_id} não encontrado."
    
    return f"Informações do usuário com ID {user_id}: {user_info[0]}"

@tool
def update_info_user(user_id: int, user_info: str, *, config: RunnableConfig) -> str:
    """Atualiza as informações do usuário na base de dados com base no user_id fornecido.
    Returns:
        Uma lista de dicionários contendo o histórico da alteração do usuário,
        a alteração que foi feita e o histórico passado dele.
    """
    
    configuration = config.get("configurable", {})
    configured_user_id = configuration.get("user_id", None)
    
    if configured_user_id is None:
        raise ValueError("Nenhum user_id configurado.")

    if user_id != configured_user_id:
        raise ValueError(f"O ID do usuário fornecido ({user_id}) não corresponde ao ID configurado ({configured_user_id}).")

    db = "caminho_para_banco_de_dados.db" #Faltando o caminho da DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM user_info WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    
    if user is None:
        raise ValueError(f"Usuário com ID {user_id} não encontrado.")

    cursor.execute(
        "UPDATE user_info SET user_info = ? WHERE user_id = ?",
        (user_info, user_id)
    )

    conn.commit()
    conn.close()
    
    return f"Informações do usuário com ID {user_id} atualizadas com sucesso."

@tool
def delete_info_user(user_id: int, *, config: RunnableConfig) -> str:
    """Deleta as informações do usuário, sem excluir seu histórico, no banco de dados com base no user_id fornecido.
    
    Returns:
        Uma lista de dicionários cada um contendo detalhes do usuário,
        associados ao seu histórico and bet details."""
    
    configuration = config.get("configurable", {})
    configured_user_id = configuration.get("user_id", None)
    
    if configured_user_id is None:
        raise ValueError("Nenhum user_id configurado.")
    
    if user_id != configured_user_id:
        raise ValueError(f"O ID do usuário fornecido ({user_id}) não corresponde ao ID configurado ({configured_user_id}).")
    
    db = "caminho_para_banco_de_dados.db" #Faltando o caminho da DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM user_info WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    
    if user is None:
        conn.close()
        raise ValueError(f"Usuário com ID {user_id} não encontrado.")

    cursor.execute("DELETE FROM user_info WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    return f"Usuário com ID {user_id} deletado com sucesso."


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)