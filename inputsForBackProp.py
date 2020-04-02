def inputsForBackProp(tics):
    inputters=[]
    outputters=[]
    for tic in tics:
        #gets the dates and the assosiated close values
        output_values = pd.read_csv('data\\training\\' + tic + '.csv')
        print(len(output_values))
        #creates an array of input vectors for a given stock and the training days
        input_values = [0]*len(output_values.index)
        data = pd.read_csv('data\\normalized_data\\' + tic + '.csv')
        data = data.set_index('date')
        for i in range(len(output_values.index)):
            date = output_values.iloc[i]['date']
            input = getInputs(tic,date,data)
            #catches error if not enough previous days
            # if input == -1:
            #     output_values.drop(date)
            input_values[i] = input
        inputters.append(input)
        outputters.append(output_values['close'].to_numpy())
    #map tics to their respective lists of inputs and outputs
    input_dict=dict(zip(tics, inputters))
    output_dict=dict(zip(tics,outputters))
    #return list of both dicts
    return [input_dict, output_dict]
    
      