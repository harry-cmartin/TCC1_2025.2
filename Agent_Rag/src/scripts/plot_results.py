import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def aggregate_data(data):
    df = pd.DataFrame(data['results'])
    
    # Calculate average time per scenario
    time_agg = df.groupby('scenario')['time_total_s'].mean().reset_index()
    
    # Calculate average relationships (Total vs LLM)
    rels_agg = df.groupby('scenario')[['total_relationships', 'llm_relationships']].mean().reset_index()
    
    # Calculate average types of relationships
    df['num_llm_types'] = df['llm_rel_types'].apply(len)
    types_agg = df.groupby('scenario')['num_llm_types'].mean().reset_index()
    
    # Calculate average density
    density_agg = df.groupby('scenario')['graph_density'].mean().reset_index()
    
    return time_agg, rels_agg, types_agg, density_agg

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
def get_scenario_names(scenarios):
    names = []
    for s in scenarios:
        if s == 'A':
            names.append('Cenário A\n(Modelo Cru)')
        elif s == 'B':
            names.append('Cenário B\n(Prompt Básico)')
        elif s == 'C':
            names.append('Cenário C\n(DSPy Otimizado)')
        elif s == 'Gabarito':
            names.append('Gabarito\n(Gold Standard)')
        else:
            names.append(s)
    return names

def plot_time(time_agg, out_dir):
    plt.figure(figsize=(8, 6))
    scenarios = get_scenario_names(time_agg['scenario'])
    ax = sns.barplot(x=scenarios, y=time_agg['time_total_s'], hue=scenarios, legend=False, palette="Blues_d")
    
    plt.title('Tempo Médio de Execução por Cenário', pad=20, fontsize=14, fontweight='bold')
    plt.ylabel('Tempo (segundos)')
    plt.xlabel('Cenário')
    
    for i, v in enumerate(time_agg['time_total_s']):
        ax.text(i, v + 0.5, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tempo_execucao.png'), dpi=300)
    plt.close()

def plot_relationships(rels_agg, out_dir):
    plt.figure(figsize=(8, 6))
    scenarios = get_scenario_names(rels_agg['scenario'])
    ax = sns.barplot(x=scenarios, y=rels_agg['llm_relationships'], hue=scenarios, legend=False, palette="Reds_d")
    
    plt.title('Arestas Semânticas Inferidas pela IA', pad=20, fontsize=14, fontweight='bold')
    plt.ylabel('Quantidade de Arestas (Média)')
    plt.xlabel('Cenário')
    
    for i, v in enumerate(rels_agg['llm_relationships']):
        ax.text(i, v + (max(rels_agg['llm_relationships']) * 0.02), f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'relacionamentos_comparacao.png'), dpi=300)
    plt.close()

def plot_relationship_types(types_agg, out_dir):
    plt.figure(figsize=(8, 6))
    scenarios = get_scenario_names(types_agg['scenario'])
    ax = sns.barplot(x=scenarios, y=types_agg['num_llm_types'], hue=scenarios, legend=False, palette="viridis")
    
    plt.title('Diversidade Semântica (Tipos de Relacionamento IA)', pad=20, fontsize=14, fontweight='bold')
    plt.ylabel('Quantidade de Tipos Distintos')
    plt.xlabel('Cenário')
    
    for i, v in enumerate(types_agg['num_llm_types']):
        ax.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'diversidade_tipos.png'), dpi=300)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(base_dir, 'results', 'geracao_nova_corrigida.json')
    gabarito_path = os.path.join(base_dir, 'results', 'gabarito_stats.json')
    out_dir = os.path.join(base_dir, 'results', 'plots')
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reading data from {json_path}...")
    data = load_data(json_path)
    
    # if os.path.exists(gabarito_path):
    #     print(f"Reading data from {gabarito_path}...")
    #     gabarito_data = load_data(gabarito_path)
    #     data['results'].extend(gabarito_data['results'])

    
    time_agg, rels_agg, types_agg, density_agg = aggregate_data(data)
    
    set_style()
    print("Generating plots...")
    plot_time(time_agg, out_dir)
    plot_relationships(rels_agg, out_dir)
    plot_relationship_types(types_agg, out_dir)
    print(f"Plots saved successfully in {out_dir}!")

if __name__ == "__main__":
    main()
