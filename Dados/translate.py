import pandas as pd
import time

df = pd.read_csv("dataset_user.csv", delimiter=';')
print(f"Total de linhas: {len(df)}")
print(df.head())

import deepl
translator = deepl.Translator("key")

# Configurações de batch
BATCH_SIZE = 50  # Traduz e salva a cada 50 linhas
translated_rows = []

try:
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE].copy()
        print(f"\nTraduzindo linhas {i+1} a {min(i+BATCH_SIZE, len(df))}...")
        
        # Traduz o batch atual
        batch["user_story_pt"] = batch["user_story"].apply(
            lambda x: translator.translate_text(x, target_lang="PT-BR").text if pd.notna(x) else ""
        )
        batch["acceptance_criteria_pt"] = batch["acceptance_criteria"].apply(
            lambda x: translator.translate_text(x, target_lang="PT-BR").text if pd.notna(x) else ""
        )
        
        translated_rows.append(batch[["user_story_pt", "acceptance_criteria_pt"]])
        
        # Salva incrementalmente
        df_partial = pd.concat(translated_rows, ignore_index=True)
        df_partial.to_csv("dataset_user_traduzido_partial.csv", sep=';', index=False)
        print(f"Batch salvo! Total traduzido: {len(df_partial)} linhas")
        
        # Pausa para evitar rate limit
        time.sleep(0.5)
    
    # Salva arquivo final
    df_final = pd.concat(translated_rows, ignore_index=True)
    df_final.to_csv("dataset_user_traduzido.csv", sep=';', index=False)
    print("\nTradução concluída com sucesso!")
    print(f"Arquivo final 'dataset_user_traduzido.csv' salvo com {len(df_final)} linhas.")
    
except Exception as e:
    print(f"\nErro na tradução: {e}")
    print(f"Progresso salvo até a linha {len(pd.concat(translated_rows, ignore_index=True)) if translated_rows else 0}")
    print("Verifique 'dataset_user_traduzido_partial.csv' para o progresso atual.")

# Imprimir o head do resultado
if translated_rows:
    df_result = pd.concat(translated_rows, ignore_index=True)
    print("\nHead do DataFrame traduzido:")
    print(df_result.head())