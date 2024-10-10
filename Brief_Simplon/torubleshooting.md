

# Scrapping


* aller sur la page internet d'un livre pour voir la structure : right click & "Inspecter"
----> il y a un tableau avec les info du livre

* utiliser openai to write code to scrap whilst providing la page html sauvergardé en local { exemple_book_html_page.html } 
----> "given the following web page that i will provide next,  write a python code that uses bs4 to scrap the table class="bibrec" and from each tr, put the text of th as column name of td as value in a dataframe"

* adjust code to deal with exeptions (try/except)


# add counter to save batches of file extracted from guttenberg...





### Logique app à déployer

on defini des agents avec une description qui permette au llm de savoir quel outil utiliser

prompt_systeme = """ on va te poser une question sur des livres, tu as a disposition 2 outils, l'un permet de retrouver des informations dans les metadonnées qui comprend ... ... ..., l'autre outil te permet de récupérer le texte en entier


def outil1:
""" à utiliser si une question generale qui est contenu dans 
"""


def outil2:
"""à utiliser si une question spécifique n'est pas trové dans le rag
"""
parcours la page internet () fait en un embedding et a partir de cet embedding retrouve les information pertinentes dasn le texte pour répondre à la quastion.




* Créer un script FastAPI : Créez un fichier Python, par exemple main.py, et configurez-le pour utiliser FastAPI et LangChain.

* Définir l'agent LangChain : Configurez votre agent LangChain dans le script.

* Créer les endpoints API : Définissez les endpoints FastAPI qui utiliseront l'agent LangChain.


Voici un exemple de code pour illustrer ces étapes :

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import Agent
from langchain.agents.tools import Tool

app = FastAPI()

# Exemple de définition d'un agent LangChain
class MyAgent(Agent):
    def __init__(self):
        self.tools = [
            Tool(name="example_tool", description="An example tool", func=self.example_tool)
        ]

    def example_tool(self, input_text: str) -> str:
        # Logique de l'outil
        return f"Processed: {input_text}"

    def run(self, input_text: str) -> str:
        # Logique de l'agent
        for tool in self.tools:
            if tool.name == "example_tool":
                return tool.func(input_text)
        return "No suitable tool found"

# Initialiser l'agent
agent = MyAgent()

# Modèle Pydantic pour la requête
class AgentRequest(BaseModel):
    input_text: str

# Endpoint pour interagir avec l'agent
@app.post("/agent/")
async def run_agent(request: AgentRequest):
    try:
        result = agent.run(request.input_text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exécuter l'application avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)