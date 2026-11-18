
# Gestion des arguments du programme
import sys

# Bibliotheque pour afficher les graphiques
import matplotlib.pyplot as plt

# Gestion des chemins de fichiers
from pathlib import Path

# Recupere la fonction de lecture du CSV
from train import load_data

# Recupere les fonctions de chargement et de prediction
from predict import load_theta


def calculate_r2(mileages, prices, theta0, theta1):
    """
    Calcule la precision du modele avec le coefficient R2.
    Compare les prix reels aux prix estimes par le modele.
    Retourne le coefficient de determination R2.
    """
    # Calcule le prix moyen des voitures
    mean_price = sum(prices) / len(prices)

    # Calcule la somme des erreurs du modele au carre
    errors = sum(
        (price - (theta0 + theta1 * km)) ** 2
        for km, price in zip(mileages, prices)
    )

    # Calcule la variation totale des prix
    total = sum((price - mean_price) ** 2 for price in prices)

    # Evite une division par zero
    if total == 0:
        return 1.0 if errors == 0 else 0.0

    # Retourne le coefficient de determination R2
    return 1 - errors / total


def plot_regression(mileages, prices, theta0, theta1):
    """
    Affiche les donnees du fichier CSV dans un graphique.
    Trace la droite de regression calculee par le modele.
    Permet de comparer les prix reels aux prix estimes.
    """
    # Calcule les prix estimes aux bornes du graphique
    x_values = [min(mileages), max(mileages)]
    # Calcule les prix avec la formule de regression lineaire
    y_values = [
        theta0 + theta1 * km for km in x_values
    ]

    # Affiche les donnees du fichier CSV
    plt.scatter(mileages, prices, label="Real prices")

    # Trace la droite de regression
    plt.plot(x_values, y_values, color="red", label="Regression line")

    # Ajoute les titres et la legende
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Linear Regression")
    plt.legend()
    plt.grid(True)

    # Ouvre le graphique
    plt.show()


def main():
    
    # Verifie que le fichier CSV est fourni
    if len(sys.argv) != 2:
        print("Usage: python visualize.py data.csv")
        sys.exit(1)
    try:
        # Verifie que le modele a ete entraine
        if not (Path(__file__).resolve().parent / "theta.json").is_file():
            raise ValueError("Model not trained. Run 'make train data.csv' first.")
        # Charge les donnees et les coefficients
        mileages, prices = load_data(sys.argv[1])
        theta0, theta1 = load_theta()

        # Calcule et affiche la precision du modele
        r2 = calculate_r2(mileages, prices, theta0, theta1)
        print(f"Model R2 score: {r2:.4f}", flush=True)

        # Affiche les donnees et la droite
        plot_regression(mileages, prices, theta0, theta1)

    except (ValueError, OSError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
