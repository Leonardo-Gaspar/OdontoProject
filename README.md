🦷 Chat Odontológico com IA (Detecção de Cáries)
# Sumário
1. Descrição do Projeto
   
2. Reflexão sobre o Desenvolvimento
   
3. Como Clonar e Executar o Projeto
   
4. Funcionalidades
   
5. Organização do Projeto
    
6. Estrutura de Pastas

# 1 - Descrição do Projeto
Este projeto consiste em um sistema web interativo, desenvolvido com Streamlit, que permite ao usuário enviar imagens odontológicas (radiografias, fotos intraorais, etc.) para análise automática de cáries utilizando um modelo de IA hospedado na plataforma Roboflow.
O sistema exibe as detecções na imagem enviada, informando a localização e a confiança de cada cárie identificada. Todas as imagens analisadas são salvas e podem ser consultadas em uma galeria acessível por outra página do app.

# 2 - Reflexão sobre o Processo de Desenvolvimento
O desenvolvimento deste projeto foi uma experiência enriquecedora, tanto do ponto de vista técnico quanto conceitual.
Ao longo do processo, enfrentei desafios que envolveram desde a integração de APIs externas (Roboflow) até a criação de uma interface amigável e funcional para o usuário, utilizando o Streamlit.

Um dos principais aprendizados foi compreender a importância de prototipar rapidamente utilizando ferramentas no-code/low-code como o Roboflow para obter modelos de IA treinados, mesmo sem profundo conhecimento em machine learning. Isso permitiu focar na entrega de valor ao usuário final, ao invés de investir tempo apenas em pesquisa e desenvolvimento de modelos do zero.

Outro ponto relevante foi a organização do código e dos dados. Implementar a funcionalidade de galeria de imagens exigiu pensar em persistência, versionamento e apresentação dos resultados, aspectos fundamentais para qualquer aplicação real de IA.

Além disso, refletir sobre a usabilidade foi essencial: tornar o app simples, intuitivo e útil, com feedbacks claros e navegação fácil, é o que diferencia um projeto técnico de uma solução realmente aplicável no cotidiano clínico ou acadêmico.

Por fim, documentar e estruturar o projeto no GitHub reforçou a importância da colaboração, reprodutibilidade e manutenção de projetos de software, habilidades cada vez mais exigidas no mercado de trabalho.

# 3 - Como Clonar e Executar o Projeto
Pré-requisitos
Python 3.8+

Conta gratuita no Roboflow

Chave de API do Roboflow

Passos

# Clone o repositório
`git clone https://github.com/Leonardo-Gaspar/OdontoProject.git`
`cd OdontoProject`

#  Crie e ative um ambiente virtual
python -m venv venv

source venv/bin/activate  # Linux/Mac

venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

# Execute o app
streamlit run app.py
Abra o navegador no endereço exibido pelo Streamlit (geralmente http://localhost:8501).

# 4 - Funcionalidades
Envio de imagens odontológicas para análise automática de cáries.

Visualização dos resultados com bounding boxes e níveis de confiança.

Galeria de imagens salvas, acessível por uma segunda página do app.

Interface simples e intuitiva (Streamlit).

Processamento via IA usando modelo treinado e hospedado no Roboflow.

# 5 - Organização do Projeto
O projeto segue uma estrutura simples e clara, facilitando a manutenção e a colaboração:

# 6 - Estrutura de Pastas
Entrega_final_ruim/
│
├── app.py                # Código principal do Streamlit
├── requirements.txt      # Dependências do projeto
├── saved_images/         # Pasta onde as imagens processadas são salvas
├── README.md             # Documentação do projeto
└── .gitignore            # Arquivos e pastas ignorados pelo Git
O código está modularizado, com funções para salvar e listar imagens, além das páginas separadas para chat e galeria.

Toda a documentação e instruções estão centralizadas neste README.md.

O diretório saved_images/ é criado automaticamente para armazenar as imagens processadas.
