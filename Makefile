
PYTHON = python3

.PHONY: all predict train bonus clean

all: predict

predict:
	$(PYTHON) predict.py

train:
	$(PYTHON) train.py

bonus:
	$(PYTHON) visualize.py

clean:
	rm -f theta.json
