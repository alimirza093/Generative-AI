from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

model = ChatMistralAI(model = "open-mistral-7b", temperature=0.7)

class ProductData(BaseModel):
    name : str
    brand : str
    category : str
    price : Optional[float]
    currency : Optional[str]
    rating : Optional[float]
    features : List[str]
    description : str
    
parser = PydanticOutputParser(pydantic_object=ProductData)

prompt = ChatPromptTemplate([
    ("system","""
        Extract useful product information from paragraph
        {formated_response} 
     """),
    ("human","{paragraph}")
])

para = input("Enter Paragraph: ")

final_prompt = prompt.invoke(
    {
        "formated_response" : parser.get_format_instructions(),
        "paragraph" : para
    }
)

response = model.invoke(final_prompt)
product_data = parser.parse(response.content)

print(product_data)