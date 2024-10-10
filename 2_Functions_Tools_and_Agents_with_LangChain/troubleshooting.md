# TEST by yourself : OpenAI FUnction Calling in LangCHain
* define pydantic object { class ArtistSearch(BaseModel): }
* convert to openai function { convert_pydantic_to_openai_function -- a list of thoose }
* bind to model { model.bind(functions=functions) } 
* test various input { .invoke / .batch / . }


# Tagging and Extraction Using OpenAI functions

### Tagging : function description => select arg from input and output strucuture json that is a function call from openai (json) == evaluate input text and generate structure output

### Extraction : from strucuture descritpion , exctract a LIST of parameters fitting a given schema

#### tagging
* cretae deterministic model (temperature=0)
* create prompt with system message (ChatPromptTemplate.from_messages([("system":"..."),("user":"{input}")]))
* force model to use function (model.bind(..., function_call={"name": "Tagging"}))
* combine prompt & model ( chain = prompt | model_with_function)
* parse the ouptup arguments to json ( chain = ... | JsonOutputFunctionsParser() )

#### Extraction
* create a parent class --extraction-- to list the tagging function outputs
* convert extraction class to openai function (not the tagging class)
* setup the model tu use the extration function with binding 
* setup the extraction chain with prompt and output (use Key output parser using key_name from extraction function)
* refine the promp

#### preprocess a long text in chunks




# Tools and Routing 

#### make function from scratch (ex wikipedia in notebook)
* define input scheme with pydantic { from pydantic import BaseModel, Field }
    -> class NameOfScheme(BaseModel):
        arg1 : float = Field(..., description="description of arg1")
        ...
* use @tool decorator to create agent { from langchain.agents import tool }
    -> @tool(args_schema=NameOfScheme)
        def Name_Of_Agent:
        """ this agent is used to do that """  --> will be use by llm to know when to use
        URL_API = "..."
        params_api = { ... }
        response = requests.get(..., params_api)
        return f"the result is {response_currated}"
====> Name_Of_Agent.name / .description / .args
* format tool to function { from langchain.tools.render import format_tool_to_openai_function }
    -> format_tool_to_openai_function(Name_Of_Agent)
    -> Name_Of_Agent( { params_api } )
* format more function and bind to model
----> functions = [format_tool_to_openai_function(f) for f in [function1, function2] ]
====> model = ChatOpenAI(temperature=0).bind(functions=functions)
* test model : model.invoke("...")

#### making agent with langchain using api specs
* langchain format api_specs { langchain.utilities.openapi import OpenAPISpec }
----> spec = OpenAPISpec.from_text(text)   / text is a json
* langchain create function { from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn }
----> pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)

#### routing agent
* you can chain a llm model and add a parser { from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser }
----> chain = prompt | model | OpenAIFunctionsAgentOutputParser()
----> result = chain.invoke({...}) 
    - if type(result) == langchain_core.agents.AgentFinish
    => result.returns_values use llm without tool
    - if type(result) == langchain_core.agents.AgentActionMessageLog
    => result.too_input must be send to tools
* defines the route function to send tool_input to tool { from langchain.schema.agent import AgentFinish }
    -> def route(result):
        if isinstance(result, AgentFinish):
            return result.return_values['output']
        else:
            tools = {
                "search_wikipedia": search_wikipedia, 
                "get_current_temperature": get_current_temperature,
            }
            return tools[result.tool].run(result.tool_input)



# Conversational agent

permet de garder en mémoire les messages precedans 