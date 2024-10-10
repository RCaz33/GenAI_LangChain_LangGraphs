# LangChain: Models, Prompts and Output Parsers

* API calls to OpenAI via LangChain
* replace custom fct "get_completion" whith class { from langchain.chat_models import ChatOpenAI }
* define temperature & model in LangChain { chat = ChatOpenAI(temperature=0.0, model=llm_model) } <<1>>
* use prompt model { from langchain.prompts import ChatPromptTemplate } <<2>>
---> keywords in template such as style / text 
* use output parser to give instruction on formating response from llm { from langchain.output_parsers import ResponseSchema, StructuredOutputParser } <<3>>
---> create several responses_schemas in a list
===> use method .format_message on response from ChatOpenAI
* combine all
---> chat = chat = ChatOpenAI(temperature=0.0, model=llm_model) <<1>>
---> prompt = ChatPromptTemplate.from_template(template=review_template_2) <<2>>
---> messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions) <<3>>
---> response = chat(messages)
---> output_dict = output_parser.parse(response.content) <<4>>

# LangChain: Memory

* instanciate conv agent { from langchain.chat_models import ChatOpenAI }
    -> llm = ChatOpenAI(temperature=0.0, model=llm_model)
* instanciate memory { from langchain.memory import ConversationBufferMemory }
    -> memory = ConversationBufferMemory()
* Create chain of thoughts { from langchain.chains import ConversationChain }
    -> conversation = ConversationChain(llm=llm, memory = memory,verbose=True)
* Define the memory buffer { from langchain.memory import ConversationBufferWindowMemory }
    -> memory = ConversationBufferWindowMemory(k=1)
    -> add memory to chain of thought { memory.save_context(...) }
* use a token base memory { from langchain.memory import ConversationTokenBufferMemory }
    -> memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
* Use summary memory { from langchain.memory import ConversationSummaryBufferMemory }
    -> memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)


# Chains in LangChain ----> create chain of thoughs

* first define llm, prompt and chain { from langchain.chains import LLMChain }
    -> chain1 = LLMChain(llm=llm, prompt=first_prompt)
* define several chain and concatenate { from langchain.chains import SimpleSequentialChain }
    -> first_prompt = ChatPromptTemplate.from_template("get product name for {input from user}")
    -> second_prompt = ChatPromptTemplate.from_template("write description for {input from chain 1}")
    -> overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True).run("stuffs to be hanfdle by the {first prompt}")
* use a router chain formatted to feed a given chain choosen by the model answer to the prompt
    ==> see example in course 


# LangChain: Q&A over Documents (using azure)

* choose the good chain { https://python.langchain.com/v0.1/docs/modules/chains/ }
----> langchain chain is { from langchain.chains import RetrievalQA } : "This chain first does a retrieval step to fetch relevant documents, then passes those documents into an LLM to generate a response."
* if AZURE, instanciate llm { from langchain.chat_models import AzureChatOpenAI }

* load document { from langchain.document_loaders import CSVLoader }
    -> docs = CSVLoader(file_path="...").load()
* embeddings { from langchain.embeddings import AzureOpenAIEmbeddings }
             { from langchain.embeddings import OpenAIEmbeddings }
    -> embeddings = OpenAIEmbeddings() { can call embeddings.embed_query("...) }
* set up a retreiver (vectostore) to fetch document { from langchain.vectorstores import DocArrayInMemorySearch }
    -> db = DocArrayInMemorySearch.from_documents(docs,embeddings)
    -> retriever = db.as_retriever()
* instanciate LLM
    -> llm = ChatOpenAI(temperature = 0.0, model=llm_model)
* all together
----> qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True)
====> qa_stuff.run("...)

##### Also can combine all these steps in ONE line
* index document in a vector store { from langchain.indexes import VectorstoreIndexCreator }
----> index = VectorstoreIndexCreator(embedding = embedding_model,
              vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])
====> response = index.query("....", llm=llm) # automatically creates embeddings of query
* customize the index : embeddings !!! vector store !!!    



# LangChain: Evaluation

* create 5 pairs of QA with lanchain { from langchain.evaluation.qa import QAGenerateChain }
----> examples = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model)).  
                  apply_and_parse( [{"doc": t} for t in data[:5]] )
====> QAGenerateChain.from_llm(ChatOpenAI(model=llm_model)).prompt.template :: shwos the prompt used to generate questions and answers
* instanciate choosen chain
    -> qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {"document_separator": "<<<<>>>>>"})

* manual evaluation with langchain.debug = True ( ca.tokens )
----> qa.run(examples[0]["query"]) -- generate context by itself --
* generate examples
----> predictions = qa.apply(examples)
* instanciate QA tester { from langchain.evaluation.qa import QAEvalChain }
====> predictions = QAEvalChain.from_llm(llm).evaluate(examples, predictions)


