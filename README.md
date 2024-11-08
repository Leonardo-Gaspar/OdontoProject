# OdontoPrev Chatbot

## 📄 Descrição

O **OdontoPrev Chatbot** é um assistente virtual desenvolvido com **Streamlit** para facilitar o acesso e a gestão de consultas odontológicas dos clientes da OdontoPrev. Este chatbot auxilia os usuários no processo de login, pré-registro de consultas e análise preliminar de imagens odontológicas para identificar possíveis problemas dentários, como cáries e fraturas.

## 🚀 Funcionalidades

- **Assistência de Login**: Orienta os usuários no processo de autenticação em suas contas OdontoPrev.
- **Pré-registro de Consultas**: Facilita o agendamento e o pré-registro de consultas odontológicas.
- **Análise de Imagens Odontológicas**: Permite o upload de imagens dentárias para uma análise preliminar, identificando possíveis problemas como cáries, fraturas e inflamações.
- **Feedback de Usuário**: Coleta feedback dos usuários sobre as respostas do assistente para aprimorar a qualidade do atendimento.
- **Upload de Documentos**: Suporta o envio de documentos odontológicos para análise detalhada.

## 🔧 Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **Streamlit**: Framework para a criação da interface web.
- **OpenAI GPT-4**: Utilizado para processamento de linguagem natural e geração de respostas.
- **Faiss-cpu**: Ferramenta para construção de bancos de dados vetoriais e similaridade de embeddings.
- **Langchain**: Conjunto de bibliotecas para gerenciar agentes e fluxos de processamento de linguagem natural.
- **LangSmith** e **LangGraph**: Extensões de Langchain para gerenciamento de dados e fluxos de chat.
- **Pydantic**: Framework para validação de dados e gerenciamento de modelos.
- **ChromaDB**: Banco de dados vetorial para armazenamento e recuperação eficiente de embeddings.
- **Pandas**: Biblioteca para manipulação e análise de dados.
- **Huggingface-hub**: Ferramenta para integração com modelos de machine learning.
- **ConversationBufferMemory**: Memória de conversa para armazenar o histórico de interação com o usuário e melhorar o contexto das respostas.

## 📦 Instalação

### 📝 Requisitos

- Python 3.7 ou superior
- Conta na OpenAI para acessar a API GPT-4
- OPENAI API KEY

### Passos de Instalação

1. **Clone o Repositório**

   ```bash
   git clone https://github.com/seu-usuario/odontoprev-chatbot.git
   cd agent_odonto_4
   Streamlit run chat_odonto.py
