from mainTest import test
from insertUtente import insert_data
from grafici import menu_visualizzazioni

def menu():
    print("Benvenuto nel menu principale!")
    print("1. Predizione dello stato depressivo di una persona (inserimento manuale)")
    print("2. Analisi, preprocessing, addestramento e previsione su file CSV di training")
    print("3. Visualizza grafici")
    print("4. Esci")

    while True:
        choice = input("Scegli un'opzione (1-4): ").strip()
        if choice == '1':
           insert_data()
        elif choice == '2':
            test()
        elif choice == '3':
            menu_visualizzazioni()
        elif choice == '4':
            print("Uscita dal programma.")
            break
        else:
            print("Opzione non valida. Riprova.")

if __name__ == "__main__":   
    menu()
