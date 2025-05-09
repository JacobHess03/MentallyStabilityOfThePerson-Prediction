from mainTest import test
from insertUtente import insert_data

def menu():
    print("Benvenuto nel menu principale!")
    print("1. Inserisci i dati dell'utente e visualizza la predizione")
    print("2. Esegui il test e genera la submission")
    print("3. Esci")

    while True:
        choice = input("Scegli un'opzione (1-3): ").strip()
        if choice == '1':
           insert_data()
        elif choice == '2':
            test()
        elif choice == '3':
            print("Uscita dal programma.")
            break
        else:
            print("Opzione non valida. Riprova.")

if __name__ == "__main__":   
    menu()
