from openai import OpenAI
import pandas as pd

# Inicializa o cliente (substitua pela sua chave real ou use variável de ambiente)
client = OpenAI(api_key="key")

# Lê o CSV
df = pd.read_csv("user_stories.csv", delimiter=';')

# Cria uma coluna vazia para embeddings com tipo object
df["embedding"] = None
df["embedding"] = df["embedding"].astype(object)  # Garante que a coluna aceite listas

# Tamanho do batch
batch_size = 50

for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch = df.iloc[start:end]

    print(f"Processando batch {start + 1} a {end}...")

    embeddings = []
    for text in batch["user_story"]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    # Atribui os embeddings um a um usando .at (funciona melhor com listas)
    for i, emb in enumerate(embeddings):
        idx = start + i
        df.at[idx, "embedding"] = str(emb)  # Converte para string para salvar no CSV

    # Salva incrementalmente após cada batch
    df.to_csv("dataset_user_embeddings.csv", sep=';', index=False)
    print(f"Batch {start + 1} a {end} concluído e salvo incrementalmente.")

# Salva o DataFrame final com os embeddings
df.to_csv("dataset_user_traduzido.csv", sep=';', index=False)
print("Arquivo final 'dataset_user_traduzido.csv' salvo com os embeddings.")