install:
	pip3 install -r requirements.txt

run:
	cd Code/Userinterface; \
	streamlit run streamlit.py

clean:
	rm -rf .venv
