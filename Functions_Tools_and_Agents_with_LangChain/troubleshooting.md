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




# Tools and Routing .. TO DO create a chain with an agent API

* converti spec d'une API (json) en fonction langchain { openapi_spec_to_openai_fn }
--->  conda install anaconda::openapi-schema-pydantic !!! fait pblm avec python 
===>  pip install openapi-pydantic
* create model with 2 functions binded to model
* create chain and add fct to 'see' LLM choice of function { chain = ... | OpenAIFunctionAgentOutputParser }
---> if a function is called, chain.invoke({...}) returns --langchain_core.agents.AgentActionMessageLog--
---> if no call of function, it returns --langchain_core.agents.AgentFinish-- with method .return_values to see the output
* defines the route function to send tool_input to tool



# Conversational agent

permet de garder en mémoire les messages precedans 