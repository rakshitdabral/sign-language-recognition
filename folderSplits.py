import splitfolders

# spliting the main dataset into train and val directory with images into 80% train and 20% validation group

splitfolders.ratio("data" , output="output" , ratio=(0.8,0.2),group_prefix=None, move=False)