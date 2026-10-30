#!/usr/bin/env python3

# Permet de lire les paramètres theta0 et theta1 enregistrés au format JSON
import json
# Permet de vérifier que le kilométrage est un nombre fini
import math

THETA_FILE = "theta.json"



def load_theta():
    """
    Charge theta0 et theta1 depuis un fichier JSON.
    Retourne (0.0, 0.0) si le fichier n'existe pas.
    """

    try:
        # Ouvre le fichier et récupère son contenu
        with open(THETA_FILE, "r") as file:
            data = json.load(file)

        # Vérifie que les données sont stockées dans un dictionnaire
        if not isinstance(data, dict):
            raise ValueError("Theta file must contain a dictionary.")

        # Vérifie que les deux paramètres sont présents
        if "theta0" not in data:
            raise ValueError("Missing theta0 parameter.")
        if "theta1" not in data:
            raise ValueError("Missing theta1 parameter.")

        # Récupère les valeurs des paramètres
        theta0 = data["theta0"]
        theta1 = data["theta1"]

        # Vérifie que les paramètres sont des nombres
        if not isinstance(theta0, (int, float)):
            raise ValueError("Theta0 must be a number.")
        if not isinstance(theta1, (int, float)):
            raise ValueError("Theta1 must be a number.")

        # Convertit les paramètres en nombres décimaux
        theta0 = float(theta0)
        theta1 = float(theta1)

        # Vérifie que les paramètres sont des nombres finis
        if not math.isfinite(theta0):
            raise ValueError("Theta0 must be finite.")
        if not math.isfinite(theta1):
            raise ValueError("Theta1 must be finite.")

        # Retourne les paramètres au main
        return theta0, theta1

    except FileNotFoundError:
        # Initialise les paramètres à zéro avant l'entraînement
        return 0.0, 0.0

    except (json.JSONDecodeError, ValueError) as error:
        # Signale un fichier JSON invalide
        raise ValueError(f"Invalid theta file: {error}") from error


def get_mileage():
    """
    Demande le kilométrage d'une voiture à l'utilisateur.
    Vérifie que la valeur est un nombre positif ou nul.
    Retourne le kilométrage sous forme de float.
    """

    while True:
        try:
            # Demande le kilométrage à l'utilisateur
            mileage = float(input("Enter car mileage: "))

            # Vérifie que le kilométrage est un nombre fini et positif ou nul
            if not math.isfinite(mileage) or mileage < 0:
                print("Error: mileage cannot be negative.")
                continue

            # Retourne le kilométrage valide au main
            return mileage

        except ValueError:
            # Gère les saisies qui ne peuvent pas être converties en nombre
            print("Error: please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            # Interrompt proprement le programme avec Ctrl+C ou Ctrl+D
            print("\nInput cancelled.")
            return None

def estimate_price(mileage, theta0, theta1):
    # Calcule le prix estimé à partir du kilométrage et des coefficients
    price = theta0 + theta1 * mileage

    # Retourne le prix estimé
    return price

def main():
    # Recupere les donnee theta (0.0 avant training)
    theta0, theta1 = load_theta()

    # Demande le kilometrage au user
    mileage = get_mileage()
    # Arrête le programme si l'utilisateur a annulé la saisie
    if mileage is None:
        return

    # Estime le prix de la voiture
    price = estimate_price(mileage, theta0, theta1)

    # Affiche la prediction
    print(f"Estimated price: {price:.2f} €")


if __name__ == "__main__":
    main()