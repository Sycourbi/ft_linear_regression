# Permet de lire les données de kilométrage et de prix dans data.csv
import csv   
# Permet d'enregistrer les coefficients theta0 et theta1 dans theta.json
import json  
# Permet de vérifier que les données et les coefficients sont des nombres finis
import math  
# Permet de vérifier l'existence du fichier CSV et de gérer les chemins
import os
# Permet de récupérer le nom du fichier CSV passé en argument
import sys  

# Nom du fichier dans lequel seront enregistrés les coefficients
THETA_FILE = "theta.json"

# Définit la vitesse d'apprentissage de la descente de gradient
LEARNING_RATE = 0.01
# Définit le nombre de mises à jour des coefficients
ITERATIONS = 10000


def load_data(filename):
    """
    Charge les kilométrages et les prix depuis un fichier CSV.
    Vérifie que le fichier et ses données sont valides.
    Retourne deux listes : mileages et prices.
    """

    # Vérifie que le fichier possède l'extension .csv
    if not filename.lower().endswith(".csv"):
        raise ValueError("File must have a .csv extension.")

    # Initialise les deux listes qui contiendront les données
    mileages = []
    prices = []

    try:
        # Ouvre le fichier CSV en lecture UTF-8, sans modifier les retours à la ligne
        # (newline="" laisse le module csv les gérer), et le ferme automatiquement grâce à with.
        with open(filename, "r", newline="", encoding="utf-8-sig") as file:

            # Lit le fichier CSV en utilisant la première ligne comme en-tête
            reader = csv.DictReader(file)
            # Vérifie que les colonnes km et price sont présentes
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty.")
            if reader.fieldnames != ["km", "price"]:
                raise ValueError("CSV must contain 'km' and 'price' columns.")

            # Parcourt chaque ligne du fichier à partir de la ligne 2
            for line_number, row in enumerate(reader, start=2):
                # Récupère les valeurs de kilométrage et de prix
                km_value = row["km"]
                price_value = row["price"]
                # Vérifie que les deux valeurs sont présentes
                if km_value is None or price_value is None:
                    raise ValueError(f"Missing data at line {line_number}.")
                # Vérifie qu'aucune des deux valeurs n'est vide ou composée uniquement d'espaces
                if km_value.strip() == "" or price_value.strip() == "":
                    raise ValueError(f"Empty value at line {line_number}.")

                # Convertit les valeurs en nombres décimaux
                try:
                    mileage = float(km_value)
                    price = float(price_value)
                except ValueError:
                    # Signale une erreur si une valeur n'est pas numérique
                    raise ValueError(
                        f"Invalid numeric value at line {line_number}."
                    )

                # Vérifie que les nombres sont finis
                if not math.isfinite(mileage) or not math.isfinite(price):
                    raise ValueError(f"Non-finite value at line {line_number}.")
                # Vérifie que les valeurs ne sont pas négatives
                if mileage < 0 or price < 0:
                    raise ValueError(f"Negative value at line {line_number}.")

                # Ajoute les valeurs valides dans leurs listes respectives
                mileages.append(mileage)
                prices.append(price)

    except FileNotFoundError:
        # Signale que le fichier demandé n'existe pas
        raise FileNotFoundError(f"CSV file not found: {filename}")

    # Vérifie que le fichier contient suffisamment de données
    if len(mileages) < 2:
        raise ValueError("CSV must contain at least two data rows.")
    # Vérifie que les kilométrages ne sont pas tous identiques
    if min(mileages) == max(mileages):
        raise ValueError("All mileages are identical.")

    # Retourne les deux listes à la fonction main
    return mileages, prices


def normalize_data(mileages):
    # Calcule le nombre total de kilométrages
    count = len(mileages)
    # Calcule la moyenne des kilométrages
    moyenne = sum(mileages) / count
    # Initialise la somme des écarts à la moyenne élevés au carré
    somme_ecarts_au_carre = 0.0

    # Parcourt chaque kilométrage pour calculer son écart à la moyenne
    for mileage in mileages:
        difference = mileage - moyenne
        somme_ecarts_au_carre += difference ** 2

    # Calcule la variance puis l'écart-type des kilométrages
    variance = somme_ecarts_au_carre / count
    ecart_type = math.sqrt(variance)

    # Vérifie que l'écart-type n'est pas nul pour éviter une division par zéro
    if ecart_type == 0:
        raise ValueError("Cannot normalize identical mileages.")

    # Initialise la liste des kilométrages normalisés
    normalized_mileages = []

    # Normalise chaque kilométrage en soustrayant la moyenne
    # puis en divisant le résultat par l'écart-type
    for mileage in mileages:
        normalized_mileage = (mileage - moyenne) / ecart_type
        normalized_mileages.append(normalized_mileage)

    # Retourne les kilométrages normalisés, la moyenne et l'écart-type
    return normalized_mileages, moyenne, ecart_type


def train_model(normalized_mileages, prices):
    # Initialise les deux coefficients du modèle à zéro
    theta0 = 0.0
    theta1 = 0.0

    # Récupère le nombre total de voitures dans le fichier CSV
    count = len(normalized_mileages)

    # Répète l'entraînement pour améliorer progressivement les coefficients
    for iteration in range(ITERATIONS):

        # Initialise les sommes des erreurs nécessaires aux deux coefficients
        somme_erreurs_theta0 = 0.0
        somme_erreurs_theta1 = 0.0

        # Parcourt toutes les voitures du fichier CSV
        for i in range(count):

            # Récupère le kilométrage normalisé et le prix réel de la voiture
            mileage = normalized_mileages[i]
            price = prices[i]

            # Calcule le prix estimé avec les coefficients actuels
            estimated_price = theta0 + theta1 * mileage

            # Calcule la différence entre le prix estimé et le prix réel
            error = estimated_price - price

            # Additionne les erreurs pour calculer la correction de theta0
            somme_erreurs_theta0 += error

            # Additionne les erreurs multipliées par le kilométrage pour theta1
            somme_erreurs_theta1 += error * mileage

        # Calcule les corrections moyennes à appliquer aux deux coefficients
        correction_theta0 = LEARNING_RATE * somme_erreurs_theta0 / count
        correction_theta1 = LEARNING_RATE * somme_erreurs_theta1 / count

        # Met à jour simultanément les deux coefficients
        theta0 = theta0 - correction_theta0
        theta1 = theta1 - correction_theta1

    # Retourne les coefficients obtenus après l'entraînement
    return theta0, theta1


def denormalize_theta(theta0, theta1, moyenne, ecart_type):
    # Adapte theta1 pour utiliser les kilomètres réels
    theta1 = theta1 / ecart_type

    # Adapte theta0 pour tenir compte de la moyenne des kilométrages
    theta0 = theta0 - theta1 * moyenne

    # Retourne les deux coefficients adaptés aux kilomètres réels
    return theta0, theta1


def main():

    # Vérifie qu'un seul nom de fichier CSV a été fourni
    if len(sys.argv) != 2:
        print("Usage: make train FILE=<filename.csv>")
        return

    # Récupère le nom du fichier CSV passé en argument
    filename = sys.argv[1]

    try:
        # Charge les données de kilométrage et de prix depuis le fichier CSV
        mileages, prices = load_data(filename)

        # Normalise les kilométrages pour faciliter la descente de gradient
        normalized_mileages, mean, scale = normalize_data(mileages)

        # Entraîne le modèle pour calculer les deux coefficients
        theta0, theta1 = train_model(normalized_mileages, prices)

        # Convertit les coefficients pour utiliser les kilométrages réels
        theta0, theta1 = denormalize_theta(theta0, theta1, mean, scale)

        # Regroupe les deux coefficients dans un dictionnaire
        data = {
            "theta0": theta0,
            "theta1": theta1
        }
        
        # Enregistre les coefficients dans le fichier theta.json
        with open("theta.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    except (ValueError, OSError, TypeError) as error:
        # Affiche l'erreur et arrête le programme
        print(f"Error: {error}")
        return

    # Affiche un message confirmant la fin de l'entraînement
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
