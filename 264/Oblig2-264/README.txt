Make sure you have all ".csv" and ".sav" files in the same folder as "Oblig2_John_Villanger.py"

Open the folder in your favorite IDE and run the python file.
You may need to install the joblib and holidays packages.
 
To change target you wish to predict for a model change the arguement for the model in the main function.
for example: k_neighbours_regression(encoded_data_holiday, data["Volum totalt"], seed, x_plot) ->
	     k_neighbours_regression(encoded_data_holiday, data["Volum til DNP"], seed, x_plot)
If a model has a different target than the lin-reg model the plotting will be messed up,
r2 score and mean squared error will not be affected by this.


Plots folder include plots of a year, week and the four different models on validation data.