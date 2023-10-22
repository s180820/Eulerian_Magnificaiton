def moving_average(series, window_size=5):
      # Convert array of integers to pandas series 
      
      # Get the window of series 
      # of observations of specified window size 
      windows = series.rolling(window_size) 
      
      # Create a series of moving 
      # averages of each window 
      moving_averages = windows.mean() 
      
      # Convert pandas series back to list 
      moving_averages_list = moving_averages.tolist() 
      final_list = moving_averages_list
      # Remove null entries from the list 
      #final_list = moving_averages_list[window_size - 1:] 
      
      return final_list