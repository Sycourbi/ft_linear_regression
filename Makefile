
# ft_linear_regression - Gestion de Python avec uv

UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
VENV := .venv
PYTHON := $(VENV)/bin/python

# Recupere le fichier passe apres "make train"
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

# Installe le projet et affiche les commandes
all: install
	@echo ""
	@echo "Projet pret !"
	@echo "  make train data.csv"
	@echo "  make predict"
	@echo "  make bonus data.csv"

# Installe uv et prepare l'environnement si necessaire
install:
	@if [ ! -x "$(UV)" ]; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1; \
	fi
	@if [ ! -x "$(PYTHON)" ]; then \
		"$(UV)" venv $(VENV) --python python3; \
	fi
	@if [ -f requirements.txt ]; then \
		"$(UV)" pip install --python $(PYTHON) -r requirements.txt --link-mode=copy; \
	fi
	@echo "Installation completed successfully."

# Entraine le modele avec le fichier CSV fourni
train:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Error: project not installed. Run 'make install' first."; \
		exit 1; \
	elif [ "$(words $(MAKECMDGOALS))" -ne 2 ]; then \
		echo "Error: please provide a CSV file."; \
		echo "Usage: make train data.csv"; \
		exit 1; \
	else \
		$(PYTHON) train.py $(ARGS); \
	fi

# Lance le programme de prediction
predict:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Error: project not installed. Run 'make install' first."; \
		exit 1; \
	else \
		$(PYTHON) predict.py; \
	fi

# Affiche les graphiques et la precision du modele
bonus:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Error: project not installed. Run 'make install' first."; \
		exit 1; \
	elif [ "$(words $(MAKECMDGOALS))" -ne 2 ]; then \
		echo "Error: please provide a CSV file."; \
		echo "Usage: make bonus data.csv"; \
		exit 1; \
	else \
		$(PYTHON) visualize.py $(ARGS); \
	fi

# Supprime les coefficients du modele
clean:
	@rm -f theta.json

# Supprime l'environnement virtuel et le cache Python
fclean: clean
	@rm -rf $(VENV) __pycache__
# Reinstalle le projet
re: fclean all

# Ignore le CSV comme cible Make
ifneq (,$(filter train bonus,$(firstword $(MAKECMDGOALS))))
.PHONY: $(ARGS)
$(ARGS):;@:
endif

.PHONY: all install train predict bonus clean fclean re
