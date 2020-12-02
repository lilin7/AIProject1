
****************************************
All submitted files and their content:
	1. Python code: All the Python code that we developed for this project. 
	2. Dataset: The dataset we collected, as well as a file "source_of_images.txt" detailing the source of each image. As the full dataset is too large to be in the submission, go to below link to download our dataset: https://drive.google.com/drive/folders/1XmEWv2rwU8I_m09c6KwFqylxyHu4GKw8?usp=sharing
	3. README: A readme.txt that lists all submitted files, instructions to run our code.
	4. Report: The project report, as detailed in the project description, in PDF format.
****************************************
How to run our code:
	!!!Attention!!!: As the dataset is too large to be submitted, we didn't put the full dataset in the code part. If you need to run our code, please first go to above mentioned link, download the full dataset, then replace the "train" folder and "test" folder with the downloaded full dataset, then you can run our code.

	run: python main.py

	If you only want to run training, only call the "train_phase()" method in train_phase.py, the train phase will be carried out, and the training model will be saved to "net.pkl" and "net_params.pkl", which could be read by later testing phase.
	If you only want to run testing, only call the "test_phase()" method in test_phase.py;

	By running "python main.py", our code will automatically generate the evaluation results provided in the report, including a table of results showing the accuracy, precision, recall and F1-measure printed in console, as well as a confusion matrix in both the console (plain version) and the pop up window (graphical version).
****************************************
