import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Modelo de embedding para vetorização de perguntas/respostas
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Defina o modelo LLM
llm = GoogleGenerativeAI(model="gemini-pro")

# Crie o template de prompt
prompt = PromptTemplate.from_template("Coloque a sua dúvida: {input}")

# Inicializa o índice FAISS
dimension = 384  # Dimensão de vetores (de acordo com o modelo 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(dimension)  # Índice FAISS para busca por similaridade de vetores

# Armazena o histórico de perguntas e respostas
questions = []
answers = []
vectors = []

# Loop contínuo para capturar e processar as dúvidas do usuário
while True:
    # Captura a entrada do usuário no terminal
    user_input = input("Digite sua dúvida (ou 'sair' para encerrar): ")
    
    # Verifica se o usuário deseja encerrar o loop
    if user_input.lower() == "sair":
        print("Encerrando o chat.")
        break

    # Verifica se o usuário quer saber sobre a primeira pergunta ou resposta
    if "primeira pergunta" in user_input.lower():
        if questions:
            print(f"Sua primeira pergunta foi: {questions[0]}")
        else:
            print("Você ainda não fez nenhuma pergunta.")
        continue

    if "primeira resposta" in user_input.lower():
        if answers:
            print(f"A minha primeira resposta foi: {answers[0]}")
        else:
            print("Eu ainda não forneci nenhuma resposta.")
        continue

    # Vetoriza a entrada do usuário
    user_input_vector = embedder.encode([user_input])

    # Busca no FAISS por perguntas anteriores semelhantes
    if len(vectors) > 0:
        D, I = index.search(user_input_vector, 1)  # Busca pelo vetor mais semelhante
        if D[0][0] < 0.5:  # Se encontrar uma correspondência com distância menor que 0.5 (limite arbitrário)
            print(f"Memória: Lembrei-me de uma pergunta anterior semelhante: {questions[I[0][0]]}")
            print(f"Resposta anterior: {answers[I[0][0]]}")
    
    # Formata o prompt com a entrada do usuário
    formatted_prompt = prompt.format(input=user_input)

    # Chame o modelo com o prompt formatado e capture a resposta
    response = llm.invoke(formatted_prompt)

    # Exibe a resposta gerada pelo modelo
    print(response)

    # Armazena a pergunta e resposta
    questions.append(user_input)
    answers.append(response)

    # Vetoriza e armazena a entrada do usuário
    vectors.append(user_input_vector)
    index.add(user_input_vector)
