

## Lang CHain Expression Language

chain = prompt | llm | StrOUtputParser()

indique que l'entre du prompt va vzers llm et llm va vers StrOutputParser()
chacun des prompt llm sont des objets de LancGhain donc il doit retrouver tout seul lorsqu'on lance des finction comme invoke

chain.invoke({"product"}:product)




#############

## BDD pour stocker les embeddings

il existe des bdd specialisé pour stocker des vecteurs, et la recherche dans ces bdd est optimisé pôur ces vecteurs ===> bonnes pratiques dans un boite est d'utiliser ces bdd spécialisées.

