# Permet de construire le chemin du fichier theta
from pathlib import Path
# Permet de lire les donnees de kilometrage et de prix dans data.csv
import csv   
# Permet d'enregistrer les coefficients theta0 et theta1 dans theta.json
import json  
# Permet de verifier que les donnees et les coefficients sont des nombres finis
import math
# Permet de recuperer le nom du fichier CSV passe en argument
import sys  

# Chemin du fichier theta dans le dossier du projet
THETA_FILE = Path(__file__).resolve().parent / "theta.json"

# Definit la vitesse d'apprentissage de la descente de gradient
LEARNING_RATE = 0.01
# Definit le nombre de mises a jour des coefficients
ITERATIONS = 10000


def load_data(filename):
    """
    Charge les kilometrages et les prix depuis un fichier CSV.
    Verifie que le fichier et ses donnees sont valides.
    Retourne deux listes : mileages et prices.
    """

    # Verifie que le fichier possede l'extension .csv
    if not filename.lower().endswith(".csv"):
        raise ValueError("File must have a .csv extension.")

    # Initialise les deux listes qui contiendront les donnees
    mileages = []
    prices = []

    try:
        # Ouvre le fichier CSV en lecture UTF-8, sans modifier les retours a la ligne
        # (newline="" laisse le module csv les gerer), et le ferme automatiquement grace a with.
        with open(filename, "r", newline="", encoding="utf-8-sig") as file:

            # Lit le fichier CSV en utilisant la premiere ligne comme en-tete
            reader = csv.DictReader(file)
            # Verifie que les colonnes km et price sont presentes
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty.")
            if reader.fieldnames != ["km", "price"]:
                raise ValueError("CSV must contain 'km' and 'price' columns.")

            # Parcourt chaque ligne du fichier a partir de la ligne 2
            for line_number, row in enumerate(reader, start=2):
                # Verifie que chaque ligne contient exactement deux valeurs
                if None in row:
                    raise ValueError(f"Too many values at line {line_number}.")
                # Recupere les valeurs de kilometrage et de prix
                km_value = row["km"]
                price_value = row["price"]
                # Verifie que les deux valeurs sont presentes
                if km_value is None or price_value is None:
                    raise ValueError(f"Missing data at line {line_number}.")
                # Verifie qu'aucune des deux valeurs n'est vide ou composee uniquement d'espaces
                if km_value.strip() == "" or price_value.strip() == "":
                    raise ValueError(f"Empty value at line {line_number}.")

                # Convertit les valeurs en nombres decimaux
                try:
                    mileage = float(km_value)
                    price = float(price_value)
                except ValueError:
                    # Signale une erreur si une valeur n'est pas numerique
                    raise ValueError(
                        f"Invalid numeric value at line {line_number}."
                    )

                # Verifie que les nombres sont finis
                if not math.isfinite(mileage) or not math.isfinite(price):
                    raise ValueError(f"Non-finite value at line {line_number}.")
                # Verifie que les valeurs ne sont pas negatives
                if mileage < 0 or price < 0:
                    raise ValueError(f"Negative value at line {line_number}.")

                # Ajoute les valeurs valides dans leurs listes respectives
                mileages.append(mileage)
                prices.append(price)

    except FileNotFoundError:
        # Signale que le fichier demande n'existe pas
        raise FileNotFoundError(f"CSV file not found: {filename}")

    # Verifie que le fichier contient suffisamment de donnees
    if len(mileages) < 2:
        raise ValueError("CSV must contain at least two data rows.")
    # Verifie que les kilometrages ne sont pas tous identiques
    if min(mileages) == max(mileages):
        raise ValueError("All mileages are identical.")

    # Retourne les deux listes a la fonction main
    return mileages, prices


def normalize_data(mileages):
    # Calcule le nombre total de kilometrages
    count = len(mileages)
    # Calcule la moyenne des kilometrages
    moyenne = sum(mileages) / count
    # Initialise la somme des ecarts a la moyenne eleves au carre
    somme_ecarts_au_carre = 0.0

    # Parcourt chaque kilometrage pour calculer son ecart a la moyenne
    for mileage in mileages:
        difference = mileage - moyenne
        somme_ecarts_au_carre += difference ** 2

    # Calcule la variance puis l'ecart-type des kilometrages
    variance = somme_ecarts_au_carre / count
    ecart_type = math.sqrt(variance)

    # Verifie que l'ecart-type n'est pas nul pour eviter une division par zero
    if ecart_type == 0:
        raise ValueError("Cannot normalize identical mileages.")

    # Initialise la liste des kilometrages normalises
    normalized_mileages = []

    # Normalise chaque kilometrage en soustrayant la moyenne
    # puis en divisant le resultat par l'ecart-type
    for mileage in mileages:
        normalized_mileage = (mileage - moyenne) / ecart_type
        normalized_mileages.append(normalized_mileage)

    # Retourne les kilometrages normalises, la moyenne et l'ecart-type
    return normalized_mileages, moyenne, ecart_type


def train_model(normalized_mileages, prices):
    # Initialise les deux coefficients du modele a zero
    theta0 = 0.0
    theta1 = 0.0

    # Recupere le nombre total de voitures dans le fichier CSV
    count = len(normalized_mileages)

    # Repete l'entraînement pour ameliorer progressivement les coefficients
    for iteration in range(ITERATIONS):

        # Initialise les sommes des erreurs necessaires aux deux coefficients
        somme_erreurs_theta0 = 0.0
        somme_erreurs_theta1 = 0.0

        # Parcourt toutes les voitures du fichier CSV
        for i in range(count):

            # Recupere le kilometrage normalise et le prix reel de la voiture
            mileage = normalized_mileages[i]
            price = prices[i]

            # Calcule le prix estime avec les coefficients actuels
            estimated_price = theta0 + theta1 * mileage

            # Calcule la difference entre le prix estime et le prix reel
            error = estimated_price - price

            # Additionne les erreurs pour calculer la correction de theta0
            somme_erreurs_theta0 += error

            # Additionne les erreurs multipliees par le kilometrage pour theta1
            somme_erreurs_theta1 += error * mileage

        # Calcule les corrections moyennes a appliquer aux deux coefficients
        correction_theta0 = LEARNING_RATE * somme_erreurs_theta0 / count
        correction_theta1 = LEARNING_RATE * somme_erreurs_theta1 / count

        # Met a jour simultanement les deux coefficients
        theta0 = theta0 - correction_theta0
        theta1 = theta1 - correction_theta1

    # Retourne les coefficients obtenus apres l'entraînement
    return theta0, theta1


def denormalize_theta(theta0, theta1, moyenne, ecart_type):
    # Adapte theta1 pour utiliser les kilometres reels
    theta1 = theta1 / ecart_type

    # Adapte theta0 pour tenir compte de la moyenne des kilometrages
    theta0 = theta0 - theta1 * moyenne

    # Retourne les deux coefficients adaptes aux kilometres reels
    return theta0, theta1


def main():
    # Verifie qu'un seul fichier CSV a ete fourni
    if len(sys.argv) != 2:
        print("Usage: python train.py data.csv")
        sys.exit(1)
    # Recupere le nom du fichier CSV passe en argument
    filename = sys.argv[1]

    try:
        # Charge les donnees de kilometrage et de prix depuis le fichier CSV
        mileages, prices = load_data(filename)

        # Normalise les kilometrages pour faciliter la descente de gradient
        normalized_mileages, mean, scale = normalize_data(mileages)

        # Entraine le modele pour calculer les deux coefficients
        theta0, theta1 = train_model(normalized_mileages, prices)

        # Convertit les coefficients pour utiliser les kilometrages reels
        theta0, theta1 = denormalize_theta(theta0, theta1, mean, scale)

        # Regroupe les deux coefficients dans un dictionnaire
        data = {
            "theta0": theta0,
            "theta1": theta1
        }
        
        # Enregistre les coefficients dans le fichier theta.json
        with open(THETA_FILE, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    except (ValueError, OSError, TypeError) as error:
        # Affiche l'erreur et arrete le programme
        print(f"Error: {error}")
        sys.exit(1)

    # Affiche un message confirmant la fin de l'entraînement
    print("Training completed successfully.")
    print("Model trained and saved to theta.json.")


if __name__ == "__main__":
    main()
