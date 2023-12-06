install:
	pip3 install -r requirements.txt

run:
	cd Code/UserInterface && streamlit run streamlit_app.py

run_validation:
	cd Code/validate_models && python3 vali.py

clean:
	rm -rf .venv
