import base64
from openai import OpenAI
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType


# Configuração da API Key (use variável de ambiente para segurança)
OPENAI_API_KEY = "API_KEY"  # Substitua pela sua chave de API OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Função para converter imagem para base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Função para descrever a imagem usando OpenAI Vision
def describe_image(image_path):
    image_base64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um assistente que descreve imagens, e retorna elementos quais elementos não fazem bem a imagem e quais objetos poderiam aparecer."},
            {"role": "user", "content": [
                {"type": "text", "text": "Descreva o que há nesta imagem, traga pontos ou elementos negativos e quais elementos poderiam ser adicionados para uma melhor apresentação da peça publicitária. E me responda em inglês."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

# Criar ferramenta para LangChain
image_tool = Tool(
    name="ImageAnalyzer",
    func=describe_image,
    description="Analisa uma imagem e descreve o que há nela, ache elementos negativos e de sugestões de elementos."
)

# Criar agente com LangChain
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
agent = initialize_agent(
    tools=[image_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Testar com uma imagem
if __name__ == "__main__":
    image_path = "imagem.jpg"  # Substitua pelo caminho da sua imagem
    response = agent.invoke(f"Descreva esta imagem: {image_path}")
    print(response)