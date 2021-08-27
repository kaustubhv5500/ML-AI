# R version
version

# Importing CSV file containing weather data
# data <- read.csv('tennis.csv')
data <- read.csv('weather.csv');
head(data);
# data <- array(data)

# Initializing hypothesis array based on length of csv data
num_attributes = dim(data)[2]-1;
hypothesis <- c(length<-(num_attributes))
for(i in 1:num_attributes){
  hypothesis[i] = '0';
}

# Initializing hypothesis to the first data row 
hypothesis <- array(data[1,1:num_attributes]);

# Find S: Finding a Maximally Specific Hypothesis
for(i in 1:dim(data)[1]){
  isequal <- data[i,num_attributes+1] == "yes";
  if(isequal){
    for(j in 1:num_attributes){
      isequal = data[i,j] == hypothesis[[j]];
      if(!isequal){
        hypothesis[[j]] = '?';
      }
      else{
        hypothesis[[j]] = data[i,j]; 
      }
    }
  }
}

# Most specific hypothesis based on the given set of training data
hypothesis

