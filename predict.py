#!/usr/bin/env python3

# Permet de construire le chemin du fichier theta
from pathlib import Path
# Permet de lire les parametres theta0 et theta1 enregistres au format JSON
import json
# Permet de verifier que le kilometrage est un nombre fini
import math
# Permet de recuperer le nom du fichier CSV passe en argument
import sys  

# Chemin du fichier theta dans le dossier du projet
THETA_FILE = Path(__file__).resolve().parent / "theta.json"



def load_theta():
    """
    Charge theta0 et theta1 depuis un fichier JSON.
    Retourne (0.0, 0.0) si le fichier n'existe pas.
    """

    try:
        # Ouvre le fichier et recupere son contenu
        with open(THETA_FILE, "r") as file:
            data = json.load(file)

        # Verifie que les donnees sont stockees dans un dictionnaire
        if not isinstance(data, dict):
            raise ValueError("Theta file must contain a dictionary.")

        # Verifie que les deux parametres sont presents
        if "theta0" not in data:
            raise ValueError("Missing theta0 parameter.")
        if "theta1" not in data:
            raise ValueError("Missing theta1 parameter.")

        # Recupere les valeurs des parametres
        theta0 = data["theta0"]
        theta1 = data["theta1"]

        # Verifie que les parametres sont des nombres et non des booleens
        if type(theta0) not in (int, float):
            raise ValueError("Theta0 must be a number.")
        if type(theta1) not in (int, float):
            raise ValueError("Theta1 must be a number.")

        # Convertit les parametres en nombres decimaux
        theta0 = float(theta0)
        theta1 = float(theta1)

        # Verifie que les parametres sont des nombres finis
        if not math.isfinite(theta0):
            raise ValueError("Theta0 must be finite.")
        if not math.isfinite(theta1):
            raise ValueError("Theta1 must be finite.")

        # Retourne les parametres au main
        return theta0, theta1

    except FileNotFoundError:
        # Si theta.json n'existe pas, initialise les coefficients a zero
        return 0.0, 0.0

    except (json.JSONDecodeError, ValueError) as error:
        # Signale un fichier JSON invalide
        raise ValueError(f"Invalid theta file: {error}") from error

    except OSError as error:
        # Signale un probleme d'acces au fichier
        raise OSError(f"Cannot read theta file: {error}") from error

def get_mileage():
    """
    Demande le kilometrage d'une voiture a l'utilisateur.
    Verifie que la valeur est un nombre positif ou nul.
    Retourne le kilometrage sous forme de float.
    """

    while True:
        try:
            # Demande le kilometrage a l'utilisateur
            mileage = float(input("Enter car mileage: "))

            # Verifie que le kilometrage est un nombre fini
            if not math.isfinite(mileage):
                print("Error: mileage must be a finite number.")
                continue

            # Verifie que le kilometrage est positif ou nul
            if mileage < 0:
                print("Error: mileage cannot be negative.")
                continue

            # Retourne le kilometrage valide au main
            return mileage

        except ValueError:
            # Gere les saisies qui ne peuvent pas etre converties en nombre
            print("Error: please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            # Interrompt proprement le programme avec Ctrl+C ou Ctrl+D
            print("\nInput cancelled.")
            return None

def estimate_price(mileage, theta0, theta1):
    # Calcule le prix estime a partir du kilometrage et des coefficients
    price = theta0 + theta1 * mileage
    # Empeche le prix de descendre sous zero
    price = max(0.0, price)
    # Retourne le prix estime
    return price

def main():
    # Recupere les coefficients et gere les erreurs de lecture
    try:
        theta0, theta1 = load_theta()
    except (ValueError, OSError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    # Demande le kilometrage au user
    mileage = get_mileage()
    # Arrete le programme si l'utilisateur a annule la saisie
    if mileage is None:
        return

    # Estime le prix de la voiture
    price = estimate_price(mileage, theta0, theta1)

    # Affiche la prediction
    print(f"Estimated price: {price:.2f} €")


if __name__ == "__main__":
    main()