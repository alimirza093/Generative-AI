from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatMistralAI(model = "open-mistral-7b", temperature=0.7)

class MovieData(BaseModel):
    title : str
    release_year : Optional[int]
    genre : List[str]
    director : str
    cast : List[str]
    rating : Optional[float]
    summary : str
    

parser = PydanticOutputParser(pydantic_object=MovieData)

prompt = ChatPromptTemplate([
    ("system","""
     Extract useful Movie information from the Paragraph
     {format_instructions} 
     """),
    ("human","{paragraph}")
])

para = input("Enter Paragraph: ")

final_prompt = prompt.invoke(
    {
        'format_instructions' : parser.get_format_instructions(),
        'paragraph' : para
    }
)

response = model.invoke(final_prompt)

movie_data = parser.parse(response.content)

print(movie_data)
