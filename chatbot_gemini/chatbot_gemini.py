import google.generativeai as genai
import os
from dotenv import load_dotenv

# variaveis de controle

# controla a primeira interação
n = "1"

# variável usada no loop do chatbot
pergunta = "sim"

# configuracao da api

load_dotenv()

chave_api = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=chave_api)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# loop principal do chatbot

while pergunta != "sair":
  
    if n == "1":

        pergunta = input("Olá, tudo bem? como posso ajudar? ")

        if pergunta != "sair":

            response = model.generate_content(pergunta)

            print(response.text)

        else:
            print("Até logo!")
            break

    n = "2"
       
    pergunta = input("Qual sua próxima pergunta? ")

    if pergunta != "sair":

        response = model.generate_content(pergunta)

        print(response.text)

    else:
        print("Até logo!")
        break