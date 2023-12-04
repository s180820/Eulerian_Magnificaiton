install:
	pip3 install -r requirements.txt

run:
	cd Code/Userinterface; \
	streamlit run main_SL.py

clean:
	rm -rf .venv
