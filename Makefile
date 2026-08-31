PYTHON ?= python3

.PHONY: test test-integration

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -p no:cacheprovider \
		-k 'not checked_in_manifest and not manifest_declared and not available_archive' \
		src/pairtoken/development/test_paired_neural.py \
		src/pairtoken/development/test_portfolio_evaluation.py \
		src/pairtoken/history/test_history_core.py \
		src/pairtoken/history/test_history_verify_evaluate_v2.py \
		src/pairtoken/confirmation/test_confirmation_governance.py

test-integration:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -p no:cacheprovider \
		src/pairtoken/development/test_paired_neural.py \
		src/pairtoken/development/test_portfolio_evaluation.py \
		src/pairtoken/history/test_history_core.py \
		src/pairtoken/history/test_history_verify_evaluate_v2.py \
		src/pairtoken/confirmation/test_confirmation_governance.py
