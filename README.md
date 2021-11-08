Shonit Gangoly 
[@ShoGan20](https://github.com/ShoGan20)   
 
  
> Requirements (I have been using these versions but you can use any):   
    `numpy 1.21.2`<br> `pandas 1.3.4` <br><br>
    
## Topic

The aim is to perform Matrix Factorization using Movie Lens data set.


## Function

1) So the code inputs the Movielens data set and splits it into training and testing data.
2) It then divides the input matrix into Users and Events
3) The function also has hyper parameters - Alpha(Learning Rate), Beta(dropping of learning rate throughout training), Lambda(Lowest Loss value), Epochs(Number of iterations the code will run)
4) It then updates user and event matrice using Gradient Descent Matrix Factorization formula
5) Calculates Mean Sqaured error using normalization of the values
6) Returns two matrices user, event
7) The predicted matrix is calculated by: user * transpose(item)

> ## Run:  
    
    1. Just run main.py file
    2. Use command 'python main.py'
    3. The downloading and unzipping of the data set is handled by the program
 
> ## Output:
    
    1. Training Mean Squared Error
    2. Testing Mean Squared Error
    3. Test Mean Absolute Error
    4. Comparison of Sample and Predicted Values
 
>## Link to Data Set
    
   Movie Lens (ml-latest-small): [http://files.grouplens.org/datasets/movielens/ml-latest-small.zip](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
