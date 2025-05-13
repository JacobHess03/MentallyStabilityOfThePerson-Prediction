import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ogni funzione restituisce un oggetto Figure

def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Gender', ax=ax, palette='viridis')
    ax.set_title('Distribuzione Genere')
    ax.set_xlabel('Genere')
    ax.set_ylabel('Conteggio')
    plt.close(fig)
    return fig


def plot_status_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Working Professional or Student', ax=ax, palette='viridis')
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
    sns.countplot(data=df, x='Depression', ax=ax, palette='coolwarm')
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
    plt.legend(title='Depressione', labels=['No', 'Sì'])
    plt.close(fig)
    return fig


def plot_suicidal_thoughts_by_depression(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x='Have you ever had suicidal thoughts ?', hue='Depression', ax=ax, palette='coolwarm')
    ax.set_title('Pensieri Suicidi vs Depressione')
    ax.set_xlabel('Pensieri Suicidi Precedenti')
    ax.set_ylabel('Conteggio')
    plt.legend(title='Depressione', labels=['No', 'Sì'])
    plt.close(fig)
    return fig


# def plot_depression_by_Region_Encoded(df):
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     if 'Region_Encoded' not in df.columns or 'Depression' not in df.columns:
#         ax.text(0.5, 0.5, "Dati per 'Region_Encoded' o 'Depression' non disponibili", 
#                 ha='center', va='center', fontsize=12)
#         ax.axis('off')
#         plt.close(fig)
#         return fig

#     Region_Encoded_counts = (
#         df
#         .groupby('Region_Encoded')['Depression']
#         .value_counts(normalize=True)
#         .unstack(fill_value=0)
#         .sort_values(by=1, ascending=False)
#         .head(15)
#     )
#     Region_Encoded_counts.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
#     ax.set_title('Incidenza di Depressione per Città (Top 15)')
#     ax.set_xlabel('Città')
#     ax.set_ylabel('Proporzione')
#     plt.xticks(rotation=45, ha='right')
#     plt.legend(title='Depressione', labels=['No', 'Sì'])
#     plt.tight_layout()
#     plt.close(fig)
#     return fig






def plot_depression_by_Region_Encoded(df):
    """
    Incidenza di Depressione per Regione (encoded → etichetta)
    """
    # mappatura code → label
    region_labels = {
        0: 'North',
        1: 'South',
        2: 'East',
        3: 'West',
        4: 'Central'
        # aggiungi qui eventuali altri codici
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'Region_Encoded' not in df.columns or 'Depression' not in df.columns:
        ax.text(0.5, 0.5,
                "Dati per 'Region_Encoded' o 'Depression' non disponibili",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.close(fig)
        return fig

    # conta proporzioni per codice
    counts = (
        df
        .groupby('Region_Encoded')['Depression']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .sort_values(by=1, ascending=False)
        .head(15)
    )

    # plot stacked bar
    counts.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')

    # sostituisci xtick con etichette
    codes = counts.index.tolist()
    labels = [region_labels.get(c, f"Codice {c}") for c in codes]
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_title('Incidenza di Depressione per Regione (Top 15)', fontsize=16)
    ax.set_xlabel('Regione', fontsize=12)
    ax.set_ylabel('Proporzione', fontsize=12)
    ax.legend(title='Depressione', labels=['No', 'Sì'])

    # aggiungi bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center')

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_depression_by_Degree_Group_Encoded_group(df):
    """
    Frequenza della Depressione per Gruppo di Laurea
    (encoded → etichetta, filtra solo occorrenze >10)
    """
    # mappatura code → label
    degree_labels = {
        0: 'Other',
        1: 'High School',
        2: 'Bachelor',
        3: 'Master',
        4: 'Doctorate'
        # aggiungi qui altri codici se necessario
    }
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'}

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'Degree_Group_Encoded' not in df.columns or 'Depression' not in df.columns:
        ax.text(0.5, 0.5,
                "Dati per 'Degree_Group_Encoded' non disponibili",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.close(fig)
        return fig

    # seleziona solo i gruppi con >10 occorrenze
    vc = df['Degree_Group_Encoded'].value_counts()
    top_codes = vc[vc > 10].index.tolist()

    if not top_codes:
        ax.text(0.5, 0.5,
                "Nessun gruppo con più di 10 occorrenze",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.close(fig)
        return fig

    # data filtered
    sub = df[df['Degree_Group_Encoded'].isin(top_codes)]

    # plot countplot
    sns.countplot(
        data=sub,
        x='Degree_Group_Encoded',
        hue='Depression',
        order=top_codes,
        palette='coolwarm',
        ax=ax
    )

    # trasforma ticks in etichette
    labels = [degree_labels.get(c, f"Codice {c}") for c in top_codes]
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_title('Frequenza della Depressione per Gruppo di Laurea (più comuni)', fontsize=16)
    ax.set_xlabel('Gruppo di Laurea', fontsize=12)
    ax.set_ylabel('Frequenza (Numero di Individui)', fontsize=12)

    # legenda con etichette di depressione
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [depression_labels[0], depression_labels[1]], title='Depressione')

    # aggiungi bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    plt.tight_layout()
    plt.close(fig)
    return fig




def plot_financial_stress_by_depression(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Depression', y='Financial Stress', ax=ax, palette='viridis')
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
    plt.legend(title='Depressione', labels=['No', 'Sì'])
    plt.close(fig)
    return fig


def plot_pearson_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Matrice di Correlazione Pearson (Numeriche)')
    plt.tight_layout()
    plt.close(fig)
    return fig



def plot_depression_by_study_satisfaction(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x='Depression', y='Study Satisfaction', ax=ax, palette='viridis')
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
    plt.legend(title='Depressione', labels=['No', 'Sì'])
    plt.tight_layout()
    plt.close(fig)
    return fig
