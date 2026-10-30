
PYTHON = python3

.PHONY: all predict train bonus clean

all: predict

train:
	@$(PYTHON) train.py $(FILE)

predict:
	@$(PYTHON) predict.py

%:
	@echo "Error: Unknown target '$@'."
	@echo "Usage: make train FILE=<filename.csv>"
	@exit 1

bonus:
	$(PYTHON) visualize.py

clean:
	rm -f theta.json
