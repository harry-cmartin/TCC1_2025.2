import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_latest_retrieval_json(results_dir):
    files = glob.glob(os.path.join(results_dir, 'retrieval_*.json'))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def aggregate_retrieval_data(data):
    records = []
    
    for q_data in data.get('results', []):
        # Gabarito
        gab = q_data.get('gabarito', {})
        records.append({
            'scenario': 'Gabarito',
            'time_s': gab.get('time_s', 0),
            'num_nodes': gab.get('num_nodes', 0),
            'score': gab.get('score', 0.0)
        })
        
        # Scenarios A, B, C
        scenarios = q_data.get('scenarios', {})
        for sc, sc_data in scenarios.items():
            records.append({
                'scenario': sc,
                'time_s': sc_data.get('time_s', 0),
                'num_nodes': sc_data.get('num_nodes', 0),
                'score': sc_data.get('score', 0.0)
            })
            
    df = pd.DataFrame(records)
    
    avg_df = df.groupby('scenario').mean().reset_index()
    # Ordem das barras
    order_map = {'Gabarito': 0, 'A': 1, 'B': 2, 'C': 3}
    avg_df['order'] = avg_df['scenario'].map(order_map)
    avg_df = avg_df.sort_values('order').drop('order', axis=1)
    
    return avg_df

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'

def get_scenario_names(scenarios):
    names = []
    for s in scenarios:
        if s == 'A':
            names.append('Cenário A\n(Cru)')
        elif s == 'B':
            names.append('Cenário B\n(Básico)')
        elif s == 'C':
            names.append('Cenário C\n(DSPy Otimizado)')
        elif s == 'Gabarito':
            names.append('Gabarito\n(Gold Standard)')
        else:
            names.append(s)
    return names

def plot_metric(df, metric_col, title, ylabel, filename, out_dir, color_palette, format_str='{:.2f}'):
    plt.figure(figsize=(9, 6))
    scenarios = get_scenario_names(df['scenario'])
    
    ax = sns.barplot(x=scenarios, y=df[metric_col], hue=scenarios, legend=False, palette=color_palette)
    
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel)
    plt.xlabel('Cenário')
    
    for i, v in enumerate(df[metric_col]):
        ax.text(i, v + (max(df[metric_col]) * 0.02), format_str.format(v), 
                ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, 'results')
    out_dir = os.path.join(results_dir, 'plots')
    
    os.makedirs(out_dir, exist_ok=True)
    
    json_path = get_latest_retrieval_json(results_dir)
    if not json_path:
        print(f"No retrieval_*.json found in {results_dir}.")
        return
        
    print(f"Reading data from {json_path}...")
    data = load_data(json_path)
    
    df_avg = aggregate_retrieval_data(data)
    
    set_style()
    print("Generating retrieval plots...")
    
    # 1. Plot Score
    plot_metric(df_avg, 'score', 'Qualidade Semântica (Score Juiz LLM)', 'Score (0.0 a 1.0)', 'retrieval_score.png', out_dir, 'Greens_d', '{:.2f}')
    
    # 2. Plot Tempo
    plot_metric(df_avg, 'time_s', 'Tempo Médio de Resposta', 'Tempo (segundos)', 'retrieval_tempo.png', out_dir, 'Blues_d', '{:.1f}s')
    
    # 3. Plot Nós
    plot_metric(df_avg, 'num_nodes', 'Volume de Contexto Exploratório (Nós)', 'Quantidade de Nós', 'retrieval_nos.png', out_dir, 'Oranges_d', '{:.1f}')
    
    print(f"Plots saved successfully in {out_dir}!")

if __name__ == "__main__":
    main()
