# Importiamo le librerie necessarie
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('Mentally/cleaned_train.csv')

# ################Distribuzione della Frequenza del Genere##############

def plot_gender_distribution(train):
    gender_labels = {0: 'Femmina', 1: 'Maschio'}
    gender_counts = train['Gender'].value_counts()
    gender_counts_labeled = gender_counts.rename(index=gender_labels)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=gender_counts_labeled.index, y=gender_counts_labeled.values, palette='viridis')
    plt.title('Distribuzione della Frequenza del Genere', fontsize=16)
    plt.xlabel('Genere', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')

    plt.tight_layout()
    plt.show()

# ################ Distribuzione della Frequenza: Studente vs Professionista

def plot_status_distribution(train):
    status_counts = train['Working Professional or Student'].value_counts()
    status_labels = {0: 'Studente', 1: 'Professionista'}
    status_counts_labeled = status_counts.rename(index=status_labels)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=status_counts_labeled.index, y=status_counts_labeled.values, palette='viridis')
    plt.title('Distribuzione della Frequenza: Studente vs Professionista', fontsize=16)
    plt.xlabel('Status', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')

    plt.tight_layout()
    plt.show()

########  Frequenza di Risposte su Pensieri Suicidi Precedenti ####################

def plot_suicidal_thoughts_distribution(suicidal_thoughts_counts):
    suicidal_thoughts_labels = {0: 'No', 1: 'Sì'}
    suicidal_thoughts_counts_labeled = suicidal_thoughts_counts.rename(index=suicidal_thoughts_labels)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=suicidal_thoughts_counts_labeled.index, y=suicidal_thoughts_counts_labeled.values, palette='viridis')
    plt.title('Frequenza di Risposte su Pensieri Suicidi Precedenti', fontsize=16)
    plt.xlabel('Risposta', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')

    plt.tight_layout()
    plt.show()


######################### Frequenza della Condizione di Depressione ################

def plot_depression_distribution(train):
    depression_counts = train['Depression'].value_counts()
    depression_labels = {0: 'No', 1: 'Sì'}
    depression_counts_labeled = depression_counts.rename(index=depression_labels)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=depression_counts_labeled.index, y=depression_counts_labeled.values, palette='viridis')
    plt.title('Frequenza della Condizione di Depressione', fontsize=16)
    plt.xlabel('Presenza di Depressione', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')

    plt.tight_layout()
    plt.show()

# ######################### Frequenza della Depressione per Genere #############################################

def plot_depression_by_gender(train):
    gender_labels = {0: 'Femmina', 1: 'Maschio'}
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'}

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=train, x='Gender', hue='Depression', palette='viridis')
    plt.title('Frequenza della Depressione per Genere', fontsize=16)
    plt.xlabel('Genere', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=[gender_labels[0], gender_labels[1]], rotation=0)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = [depression_labels[int(lbl)] for lbl in labels]
    ax.legend(handles, new_labels, title='Condizione Depressione')

    for container in ax.containers:
        for p in container.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width() / 2., height + 50,
                        f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


###################### Frequenza dei Pensieri Suicidi Precedenti in base alla Depressione #####

def plot_suicidal_thoughts_by_depression(train):
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'}
    suicidal_thoughts_labels = {0: 'No (Pensieri)', 1: 'Sì (Pensieri)'}

    plt.figure(figsize=(9, 6))
    ax = sns.countplot(data=train, x='Depression', hue='Have you ever had suicidal thoughts ?', palette='pastel')
    plt.title('Frequenza dei Pensieri Suicidi Precedenti in base alla Depressione', fontsize=16)
    plt.xlabel('Condizione di Depressione', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=[depression_labels[0], depression_labels[1]], rotation=0)

    handles, labels = ax.get_legend_handles_labels()
    new_legend_labels = [suicidal_thoughts_labels[int(lbl)] for lbl in labels]
    ax.legend(handles, new_legend_labels, title='Pensieri Suicidi Precedenti')

    for container in ax.containers:
        for p in container.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width() / 2., height + 50, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


################### Frequenza della Depressione per Regione

# 0     Central India              0
# 1        East India              1
# 2  North-East India              2
# 3  North-West India              3
# 4             Other              4
# 5       South India              5
# 6      West-Gujarat              6
# 7  West-Maharashtra              7
def plot_depression_by_region(train):
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'}
    region_codes_order_sorted = sorted(train['Region_Encoded'].unique())
    region_labels = {
        0: 'Central India',
        1: 'East India',
        2: 'North-East India',
        3: 'North-West India',
        4: 'Other',
        5: 'South India',
        6: 'West-Gujarat',
        7: 'West-Maharashtra'
    }

    # Create the countplot
    plt.figure(figsize=(14, 7))
    ax = sns.countplot(
        data=train,
        x='Region_Encoded',
        hue='Depression',
        palette='viridis',
        order=region_codes_order_sorted
    )

    # Replace numeric x-tick labels with region names
    xtick_names = [region_labels[code] for code in region_codes_order_sorted]
    ax.set_xticklabels(xtick_names, rotation=45, ha='right')

    # Titles and labels
    plt.title('Frequenza della Depressione per Regione', fontsize=16)
    plt.xlabel('Regione', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_legend_labels = [depression_labels[int(lbl)] for lbl in labels]
    ax.legend(handles, new_legend_labels, title='Condizione Depressione')

    plt.tight_layout()
    plt.show()




############## Distribuzione dello Stress Finanziario per Stato di Depressione
    
def plot_financial_stress_by_depression(train):
    depression_labels_plot = {0: 'No Depressione', 1: 'Sì Depressione'}

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=train, x='Depression', y='Financial Stress', palette='viridis')
    plt.title('Distribuzione dello Stress Finanziario per Stato di Depressione', fontsize=16)
    plt.xlabel('Condizione di Depressione', fontsize=12)
    plt.ylabel('Stress Finanziario', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=[depression_labels_plot[0], depression_labels_plot[1]], rotation=0)
    plt.tight_layout()
    plt.show()

############# Distribuzione dell'Età per Stato di Depressione ##############
def plot_age_distribution_by_depression(train):
    depression_labels_plot = {0: 'No Depressione', 1: 'Sì Depressione'}

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=train, x='Depression', y='Age', palette='viridis')
    plt.title("Distribuzione dell'Età per Stato di Depressione", fontsize=16)
    plt.xlabel('Condizione di Depressione', fontsize=12)
    plt.ylabel('Età', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=[depression_labels_plot[0], depression_labels_plot[1]], rotation=0)

    plt.tight_layout()
    plt.show()
   
############ Frequenza della Depressione per Gruppo di Laurea
 
def plot_depression_by_degree_group(train):
    """
    Plotta la frequenza della depressione suddivisa per gruppo di laurea utilizzando le etichette testuali.
    """
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'}
    degree_order = sorted(train['Degree_Group_Encoded'].unique())
    degree_labels = {
        0: 'Other',
        1: 'High School',
        2: 'Bachelor',
        3: 'Master',
        4: 'Doctorate'
    }

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=train,
        x='Degree_Group_Encoded',
        hue='Depression',
        palette='viridis',
        order=degree_order
    )

    # Sostituisci i codici dei gruppi di laurea con le etichette testuali
    xtick_names = [degree_labels[code] for code in degree_order]
    ax.set_xticklabels(xtick_names, rotation=0, ha='center')

    # Titolo e assi
    plt.title('Frequenza della Depressione per Gruppo di Laurea', fontsize=16)
    plt.xlabel('Gruppo di Laurea', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)

    # Legenda con etichette di depressione testuali
    handles, labels = ax.get_legend_handles_labels()
    new_legend_labels = [depression_labels[int(lbl)] for lbl in labels]
    ax.legend(handles, new_legend_labels, title='Condizione Depressione')

    # Aggiungi etichette sui bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    plt.tight_layout()
    plt.show()


############ Frequenza della Depressione per Livello di Soddisfazione nello Studio

def plot_depression_by_study_satisfaction(train):
    depression_labels = {0: 'No Depressione', 1: 'Sì Depressione'}
    satisfaction_order = sorted(train['Study Satisfaction'].unique())

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=train, x='Study Satisfaction', hue='Depression',
                       palette='viridis', order=satisfaction_order)

    plt.title('Frequenza della Depressione per Livello di Soddisfazione nello Studio', fontsize=16)
    plt.xlabel('Soddisfazione nello Studio (Scala)', fontsize=12)
    plt.ylabel('Frequenza (Numero di Individui)', fontsize=12)
    plt.xticks(rotation=0)

    handles, labels = ax.get_legend_handles_labels()
    new_legend_labels = [depression_labels[int(lbl)] for lbl in labels]
    ax.legend(handles, new_legend_labels, title='Condizione Depressione')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)

    plt.tight_layout()
    plt.show()

    
######### MATRICE DI CORRELAZIONE ##########
def plot_pearson_correlation(train):
    cols = ['Age', 'CGPA', 'Sleep Duration', 'Work/Study Hours']
    corr_matrix = train[cols].corr(method='pearson')

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', square=True,
                linewidths=0.5, linecolor='white', cbar_kws={'shrink': .75})
    plt.title('Matrice di Correlazione di Pearson', fontsize=16)
    plt.tight_layout()
    plt.show()

'''
Valori unici per ogni colonna:
------------------------------
Colonna: Gender ✅
[0 1]
------------------------------
Colonna: Age
[49. 26. 33. 22. 30. 59. 47. 38. 24. 42. 55. 51. 39. 29. 50. 23. 56. 45.
 37. 46. 31. 19. 28. 25. 41. 60. 18. 36. 21. 58. 44. 43. 40. 35. 54. 27.
 52. 48. 57. 53. 34. 20. 32.]
------------------------------
Colonna: Working Professional or Student ✅
[1 0]
------------------------------
Colonna: Academic Pressure
[0. 5. 2. 3. 4. 1.]
------------------------------
Colonna: Work Pressure
[5. 4. 0. 1. 2. 3.]
------------------------------
Colonna: CGPA
[ 0.      8.97    5.9     7.03    5.59    8.13    5.7     9.54    8.04
  9.79    8.38    6.1     7.04    8.52    5.64    8.58    6.51    7.25
  7.83    9.93    8.74    6.73    5.57    8.59    7.1     6.08    5.74
  9.86    6.7     6.21    5.87    6.37    9.72    5.88    9.56    6.99
  5.24    9.21    7.85    6.95    5.86    7.92    9.66    8.94    9.71
  7.87    5.6     7.9     5.46    6.79    8.7     7.38    8.5     7.09
  9.82    8.89    7.94    9.11    6.75    7.53    9.49    9.01    7.64
  5.27    6.      9.44    5.75    7.51    9.05    6.38    8.95    9.88
  5.32    6.27    7.7     8.1     9.59    8.96    5.51    7.43    8.79
  9.95    5.37    6.86    8.32    9.74    5.66    7.48    8.23    8.81
  6.03    5.56    5.68    5.14    7.61    6.17    8.17    9.87    8.75
  6.16    9.5     7.99    5.67    8.92    6.19    5.76    6.25    5.11
  5.58    5.65    9.89    8.03    6.61    9.41    8.64    7.21    8.28
  6.04    9.13    8.08    9.96    5.12    8.35    7.07    9.6     9.24
  8.54    8.78    8.93    8.91    9.04    6.83    5.85    7.74    6.41
  8.9     7.75    7.88    5.42    7.52    7.68    8.4     9.39    6.84
  5.99    8.62    8.53    7.47    6.78    6.42    9.92    8.39    5.89
  7.22    6.81    9.02    9.97    9.63    9.67    5.41    7.27    6.05
  6.85    9.33    5.81    6.53    5.98    6.02    6.74    5.26    7.72
  7.39    8.43    9.34    5.44    5.82    5.72    8.19    8.44    8.98
  9.37    5.8     7.28    7.6     7.91    9.17    7.46    9.43    9.91
  9.36    5.16    7.08    9.26    8.83   10.      7.8     9.46    6.63
  7.24    6.47    7.77    5.06    7.17    8.24    6.88    9.03    5.08
  5.45    8.46    9.19    6.36    8.73    7.11    9.12    9.4     8.11
  9.98    5.55    8.61    8.14    6.89    9.84    5.48    8.21    7.82
  8.55    5.79    8.77    8.29    6.92    7.37    9.7     6.26    7.26
  7.5     6.82    7.15    5.77    5.91    5.1     7.71    9.06    5.71
  5.84    9.42    6.23    6.29    5.25    9.69    9.9     6.39    8.09
  5.83    5.47    6.56    8.71    9.94    6.69    5.52    7.3     7.02
  6.33    8.07    8.37    8.      7.79    8.65    6.28    7.35    8.69
  7.12    7.32    7.13    5.97    5.09    6.91    6.76    6.52    7.45
  8.56    6.5     8.63    8.27    8.49    6.59    9.29    5.3     7.06
  5.38    6.65    9.16    8.01    8.25    8.02    8.47    7.34    8.88
  7.14    8.42    5.17    9.1     7.49    9.85    7.42    9.31    6.35
  7.      5.39    5.61    9.78    9.25    5.69    9.47    8.16    7.23
  6.46    8.26    6.32    6.77    8.85    5.03    7.65    5.78    6.24
  5.35    6.06    7.78    6.64    7.0625  6.98    6.44    6.09  ]
------------------------------
Colonna: Study Satisfaction 
[0. 2. 5. 3. 4. 1.]
------------------------------
Colonna: Job Satisfaction
[2. 3. 0. 1. 5. 4.]
------------------------------

Colonna: Sleep Duration
[ 9.   4.   5.5  7.5  1.5  7.   5.   6.5 10.5  8.5 42.5 10.   2.5  3.5
 60.5  4.5  2.  45.  35.5  8.  49.  46.5]
------------------------------
Colonna: Dietary Habits
[2. 0. 1.]
------------------------------
Colonna: Have you ever had suicidal thoughts ?
[0 1] ✅
------------------------------
Colonna: Work/Study Hours
[ 1.  7.  3. 10.  9.  6.  8.  2.  0.  5. 12.  4. 11.]
------------------------------
Colonna: Financial Stress
[2. 3. 1. 4. 5.]
------------------------------
Colonna: Family History of Mental Illness
[0 1] ✅
------------------------------
Colonna: Depression
[0 1] ✅
------------------------------
Colonna: Degree_Group_Encoded
[2. 0. 3. 4. 1.]
------------------------------
Colonna: Professional_Group_Encoded
[ 3  5 14  1  7 15 18 10 13  9  2  6 12  8  0  4 11 19 17 16]
------------------------------
Colonna: Region_Encoded 
[3 2 1 7 6 5 0 4] ✅
------------------------
'''

def menu_visualizzazioni(train):
    while True:
        print("\n📊 MENU VISUALIZZAZIONI 📊")
        print("1. Distribuzione del Genere")
        print("2. Distribuzione Studente vs Professionista")
        print("3. Frequenza Pensieri Suicidi Precedenti")
        print("4. Frequenza della Condizione di Depressione")
        print("5. Depressione per Genere")
        print("6. Pensieri Suicidi in base alla Depressione")
        print("7. Depressione per Regione")
        print("8. Stress Finanziario per Stato di Depressione")
        print("9. Età per Stato di Depressione")
        print("10. Correlazione Pearson (Age, CGPA, Sleep, Work Hours)")
        print("11. Depressione per Gruppo di Laurea")
        print("12. Depressione per Soddisfazione nello Studio")
        print("0. Esci")

        scelta = input("\nSeleziona un'opzione (0-12): ")

        if scelta == '1':
            plot_gender_distribution(train)
        elif scelta == '2':
            plot_status_distribution(train)
        elif scelta == '3':
            suicidal_thoughts_counts = train['Have you ever had suicidal thoughts ?'].value_counts()
            plot_suicidal_thoughts_distribution(suicidal_thoughts_counts)
        elif scelta == '4':
            plot_depression_distribution(train)
        elif scelta == '5':
            plot_depression_by_gender(train)
        elif scelta == '6':
            plot_suicidal_thoughts_by_depression(train)
        elif scelta == '7':
            plot_depression_by_region(train)
        elif scelta == '8':
            plot_financial_stress_by_depression(train)
        elif scelta == '9':
            plot_age_distribution_by_depression(train)
        elif scelta == '10':
            plot_pearson_correlation(train)
        elif scelta == '11':
            plot_depression_by_degree_group(train)
        elif scelta == '12':
            plot_depression_by_study_satisfaction(train)
        elif scelta == '0':
            print("Uscita dal menù.")
            break
        else:
            print("❌ Scelta non valida. Riprova.")

menu_visualizzazioni(train)