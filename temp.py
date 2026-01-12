text = "tushar is learning AI AND ML, he is very excited about the future of technology."
chunk size = 15
{
  "chunks": [
    "tushar is", ->  embeddings=[12.3,24.5,213]
    " learning ", -> embeddings=[12.3,24.5,213]
    "AI AND ML,",
    " he is ve",
    "ry excited ",
    "about the ",
    "future of ",
    "technology."
  ]
}
    " he is ve",
    "ry excited ",
    "about the ",
    "future of ",
    "technology."
  ]
}

x2221aa
scsax1

embeddings

docs->llm->answers

docs->chunks->embeddings->vectorDB->similarity search->llm->answers


#sentence transformer takes text input and converts it into numerical vectors called embeddings. These embeddings capture the semantic meaning of the text, allowing for better understanding and comparison of different pieces of text in various applications such as search, recommendation systems, and clustering.