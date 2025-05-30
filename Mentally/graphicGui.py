import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ogni funzione restituisce un oggetto Figure

def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Gender', hue='Gender', ax=ax, palette='viridis', legend=False)
    ax.set_title('Distribuzione Genere')
    ax.set_xlabel('Genere')
    ax.set_ylabel('Conteggio')
    plt.close(fig)
    return fig


def plot_status_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Working Professional or Student', hue='Working Professional or Student', ax=ax, palette='viridis', legend=False)
    ax.set_title('Distribuzione Studente vs Professionista')
    ax.set_xlabel('Status')
    ax.set_ylabel('Conteggio')
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_suicidal_thoughts_distribution(counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_title('Distribuzione Pensieri Suicidi Precedenti')
    ax.set_xlabel('Ha avuto pensieri suicidi?')
    ax.set_ylabel('Conteggio')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_depression_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Depression', hue='Depression', ax=ax, palette='coolwarm', legend=False)
    ax.set_title('Distribuzione Condizione di Depressione')
    ax.set_xlabel('Depressione (0=No, 1=Sì)')
    ax.set_ylabel('Conteggio')
    plt.xticks([0, 1], ['No', 'Sì'])
    plt.close(fig)
    return fig


def plot_depression_by_gender(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x='Gender', hue='Depression', ax=ax, palette='coolwarm')
    ax.set_title('Depressione per Genere')
    ax.set_xlabel('Genere')
    ax.set_ylabel('Conteggio')
    
    handles, labels = ax.get_legend_handles_labels()
    if handles: 
        if len(handles) == 2: # Assumendo 0='No', 1='Sì'
            ax.legend(handles, ['No', 'Sì'], title='Depressione')
        else: # Altrimenti, usa le etichette generate da Seaborn
            ax.legend(handles=handles, labels=labels, title='Depressione')
    plt.close(fig)
    return fig


def plot_suicidal_thoughts_by_depression(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x='Have you ever had suicidal thoughts ?', hue='Depression', ax=ax, palette='coolwarm')
    ax.set_title('Pensieri Suicidi vs Depressione')
    ax.set_xlabel('Pensieri Suicidi Precedenti')
    ax.set_ylabel('Conteggio')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if len(handles) == 2:
            ax.legend(handles, ['No', 'Sì'], title='Depressione')
        else:
            ax.legend(handles=handles, labels=labels, title='Depressione')
    plt.close(fig)
    return fig


def plot_depression_by_Region_Encoded(df):
    """
    Incidenza di Depressione per Regione (encoded → etichetta)
    """
    region_labels = {
        0: 'North', 1: 'South', 2: 'East', 3: 'West', 4: 'Central'
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'Region_Encoded' not in df.columns or 'Depression' not in df.columns:
        ax.text(0.5, 0.5, "Dati per 'Region_Encoded' o 'Depression' non disponibili",
                ha='center', va='center', fontsize=12)
        ax.axis('off'); plt.close(fig); return fig

    counts = (df.groupby('Region_Encoded')['Depression']
              .value_counts(normalize=True).unstack(fill_value=0)
              .sort_values(by=1, ascending=False).head(15)) # Assumendo che la colonna 1 sia 'Sì Depressione'
    
    if counts.empty:
        ax.text(0.5, 0.5, "Nessun dato da visualizzare per le regioni.",
                ha='center', va='center', fontsize=12)
        ax.axis('off'); plt.close(fig); return fig

    counts.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
    
    bar_codes = counts.index.tolist()
    tick_display_labels = [region_labels.get(c, f"Codice {c}") for c in bar_codes]

    ax.set_xticks(np.arange(len(bar_codes)))
    ax.set_xticklabels(tick_display_labels, rotation=45, ha='right')
    
    ax.set_title('Incidenza di Depressione per Regione (Top 15)', fontsize=16)
    ax.set_xlabel('Regione', fontsize=12); ax.set_ylabel('Proporzione', fontsize=12)
    
    if 0 in counts.columns and 1 in counts.columns:
         ax.legend(title='Depressione', labels=['No', 'Sì']) 
    else: # Fallback generico
        current_handles, current_labels = ax.get_legend_handles_labels()
        if current_handles:
            ax.legend(handles=current_handles, labels=current_labels, title='Depressione')


    for c_idx, c in enumerate(ax.containers):
        text_color = 'white'
        labels_for_bar = [f'{h:.0%}' if (h := bar.get_height()) > 0.02 else '' for bar in c]
        ax.bar_label(c, labels=labels_for_bar, label_type='center', color=text_color, fontsize=8, fontweight='bold')
    plt.tight_layout(); plt.close(fig); return fig


def plot_depression_by_Degree_Group_Encoded_group(df):
    """
    Frequenza della Depressione per Gruppo di Laurea
    """
    degree_labels = {0: 'Other', 1: 'High School', 2: 'Bachelor', 3: 'Master', 4: 'Doctorate'}
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'} # Usato per la legenda
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'Degree_Group_Encoded' not in df.columns or 'Depression' not in df.columns:
        ax.text(0.5, 0.5, "Dati per 'Degree_Group_Encoded' non disponibili",
                ha='center', va='center', fontsize=12)
        ax.axis('off'); plt.close(fig); return fig

    vc = df['Degree_Group_Encoded'].value_counts()
    top_codes = sorted(vc[vc > 10].index.tolist())
    if not top_codes:
        ax.text(0.5, 0.5, "Nessun gruppo con più di 10 occorrenze",
                ha='center', va='center', fontsize=12)
        ax.axis('off'); plt.close(fig); return fig

    sub = df[df['Degree_Group_Encoded'].isin(top_codes)]
    sns.countplot(data=sub, x='Degree_Group_Encoded', hue='Depression',
                  order=top_codes, palette='coolwarm', ax=ax)
    
    tick_display_labels = [degree_labels.get(c, f"Codice {c}") for c in top_codes]
    ax.set_xticks(np.arange(len(top_codes)))
    ax.set_xticklabels(tick_display_labels, rotation=45, ha='right')
    
    ax.set_title('Frequenza della Depressione per Gruppo di Laurea (più comuni)', fontsize=16)
    ax.set_xlabel('Gruppo di Laurea', fontsize=12)
    ax.set_ylabel('Frequenza (Numero di Individui)', fontsize=12)
    
    handles, current_labels = ax.get_legend_handles_labels()
    if handles: 
        try:
            mapped_labels = [depression_labels[int(l)] for l in current_labels]
            ax.legend(handles, mapped_labels, title='Depressione')
        except (ValueError, KeyError): 
            if len(handles) == 2:
                ax.legend(handles, ['No', 'Sì'], title='Depressione') # Fallback generico
            else:
                ax.legend(handles=handles, labels=current_labels, title='Depressione')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)
    plt.tight_layout(); plt.close(fig); return fig


def plot_financial_stress_by_depression(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Depression', y='Financial Stress', hue='Depression', ax=ax, palette='viridis', legend=False)
    ax.set_title('Stress Finanziario vs Depressione')
    ax.set_xlabel('Depressione (0=No, 1=Sì)')
    ax.set_ylabel('Stress Finanziario')
    plt.xticks([0, 1], ['No', 'Sì'])
    plt.close(fig)
    return fig


def plot_age_distribution_by_depression(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Depression', kde=True, ax=ax, palette='coolwarm')
    ax.set_title('Distribuzione Età per Condizione di Depressione')
    ax.set_xlabel('Età')
    ax.set_ylabel('Frequenza')

    handles, labels = ax.get_legend_handles_labels()
    if handles: 
        if len(handles) == 2: # Assumendo 0='No', 1='Sì'
            ax.legend(handles, ['No', 'Sì'], title='Depressione')
        else:
            ax.legend(handles=handles, labels=labels, title='Depressione')
    plt.close(fig)
    return fig


def plot_pearson_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty or numeric_df.shape[1] < 2 :
        ax.text(0.5, 0.5, "Dati numerici insufficienti per la correlazione.",
                ha='center', va='center', fontsize=12)
        ax.axis('off'); plt.close(fig); return fig
        
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=.5)
    ax.set_title('Matrice di Correlazione Pearson (Numeriche)')
    plt.tight_layout(); plt.close(fig); return fig


def plot_depression_by_study_satisfaction(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x='Depression', y='Study Satisfaction', hue='Depression', ax=ax, palette='viridis', legend=False)
    ax.set_title('Soddisfazione Studio vs Depressione')
    ax.set_xlabel('Depressione (0=No, 1=Sì)')
    ax.set_ylabel('Soddisfazione Studio')
    plt.xticks([0, 1], ['No', 'Sì'])
    plt.close(fig)
    return fig


def plot_depression_by_status(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x='Working Professional or Student', hue='Depression', ax=ax, palette='coolwarm')
    ax.set_title('Depressione per Status (Studente/Professionista)')
    ax.set_xlabel('Status')
    ax.set_ylabel('Conteggio')
    plt.xticks(rotation=45, ha='right') 
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        if len(handles) == 2:
            ax.legend(handles, ['No', 'Sì'], title='Depressione')
        else:
            ax.legend(handles=handles, labels=labels, title='Depressione')
    plt.tight_layout(); plt.close(fig); return fig