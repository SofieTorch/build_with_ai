from langchain import PromptTemplate
from langchain_openai import ChatOpenAI

OPENAI_API_KEY="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"

def get_antithesis(input):
    template = """Estoy buscando la antítesis de la siguiente afirmación. La antítesis es una declaración que es totalmente opuesta a la afirmación original. Genera una respuesta que sea completamente opuesta a la afirmación dada.
    Afirmacion: {statement}
    Antithesis: """ 

    promp_template = PromptTemplate(
        input_variables=["statement"],
        template=template
    )
    formatted_prompt = promp_template.format(statement=input)
    
    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
    
    result = llm.invoke(formatted_prompt)
    parsed_result = result.content  
    
    return parsed_result
