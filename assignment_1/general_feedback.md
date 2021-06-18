# General feedback

## Cross-validation (CV)

- Some of you forgot to split the data we gave to you intro training and test datasets. When someone asks you to compute the TEST performance (or to estimate the generalization ability) of a model, unless the data given to you is already split into training and test datasets, you need to split the data into training and test datasets (which should NOT overlap, although this may not always be the case or even possible; so, you should not have used all the data we gave to you for training and then pick a subset of this training data for testing), use the training data for training and the test data for testing. This idea of cross-validation is fundamental in machine learning, so not doing CV is a big mistake (although, in the bonus task 3, we said we would have then tested your model on a different separate test set).

## k-fold cross-validation

- There's also k-fold cross-validation (which some of you used), where you try to reuse all data for both training and testing, but, at each iteration, you keep the test data separate from the training data.

- Some of you tried to train, validate and test the models with different permutations of the original dataset given to you to have an average estimate of the test performance, but note that this is not exactly how k-fold cross validation works.

## Retraining

- Some of you tried to reevaluate the performance over the entire dataset (without first fully retraining the model on entire dataset) 

    - In class, we have seen that, once you have chosen a model and tested it on the test set, you should retrain this model over the entire dataset, because, in expectation, the more data you have the better your model should become. But note that, once you get this retrained model, if you compute any performance metric using the same data you used for retraining, it will be a biased value (same idea of why we use a separate test set for testing)

## Data for non-linear model

- You were not required to add the features added for task 1 in task 2, i.e. you should have just passed observations with the original 2 features to the non-linear model.

## Data visualization

- Some of you tried to visualise the shape of the data to understand whether it's linearly separable or not, and this was a good idea.

## Expected risk can be decomposed into approximation risk, estimation risk and inherent risk

- Some of you did not really understand what the approximation risk is. 

- Remember that the expected risk can be decomposed into 

    1. the approximation risk (which is the risk associated with the choice of the model family, also known as "hypothesis class" in learning theory, and it does not depend on the training data, so it remains constant if you don't change the model family, e.g. the architecture of the neural network), 
  
    2. estimation risk (which is the risk associated with the choice of function that you chose FROM your model family, i.e. if you chose a function that is too bad, this risk will be high; this risk depends both on the model family and training data)
  
    3. inherent risk (which is related to the irreducible error)

- Generally, simpler model families are more likely to lead to high approximation risk, so the correct answer to question "Which region is associated with high risk?" is section (a), where the model complexity is low; in fact, the concept of approximation risk is related to the concept of under-fitting. Maybe this answer https://ai.stackexchange.com/a/23887/2444 could be helpful.

- Many of you said that the expected risk can not be brought to zero, but you didn't give the main reason: i.e. because of the inherent risk (in fact, this is a lower bound on the expected risk). 

- Some of you pointed that, if you increase the model complexity, we may have over-fitting, so, the expected/structural risk, which is the generalisation ability of a model, would increase. However, note that, with an infinitely big dataset, you wouldn't need more data to learn the actual function. Moreover, with an infinitely big model family, the approximation risk would be minimal. So, in theory, it's possible to decrease the expected risk if you increase the training data and the complexity of the hypothesis class, and what remains is the inherent risk. However, it's true that, with a fixed training dataset, if you increase the complexity of the hypothesis class, you may have over-fitting.

## Hypothesis testing, null hypothesis, p-value and confidence intervals

- Not all of you used a statistical test, as we saw during the lectures and in one lab, to determine which model is "statistically better". We were expecting you to do this (both in task 2 and 3) and the "statistically" part was a hint, so that you are at least familiar with the idea of a statistical test, although we have not gone into the details of statistical tests.

    - To assess which model is statistically better, some of you used the R^2 metric, which I don't think we have seen during the lectures, although it's mentioned in the chapter on linear regression of one of the textbooks of the course

- Be aware of the fact there are other hypothesis tests apart from the ones we have seen during the lectures; to use a specific hypothesis test, you need to make some assumptions;
    
    - one important thing when using a hypothesis test is to clearly define what the "null hypothesis" is; the alternative hypothesis would just be the complement

    - then you also need to define a so-called "statistical level" (often denoted by the Greek letter "alpha"), which is a threshold probability that allows you to reject the null hypothesis (and we say with "statistical significance"); typical values of alpha are 0.01 or 0.05;
        - Almost no one did this!

    - You can think of the p-value as a "conditional probability" where the condition is that the null hypothesis is true, so we can write it as follows p-value = P(test statistic in some range given that null hypothesis is true), 
    
        - (However, note that some people will claim that this definition of the p-value is wrong because, when dealing with p-values we are in the realm of "[frequentist statistics](https://stats.stackexchange.com/q/22)", so we cannot condition on the null hypothesis, which would not be a random variable (but could be a random variable in Bayesian statistics): see here https://normaldeviate.wordpress.com/2013/03/14/double-misunderstandings-about-p-values/ or here https://theoreticalecology.wordpress.com/2013/03/15/is-the-p-value-a-conditional-probability/. But, for simplicity, you can ignore this.)

        - (What is definitely not true is that the p-value is the probability the the null hypothesis is true: this is wrong!)

    - after having performed your hypothesis test, if the p-value < alpha, you reject the null hypothesis; this does not mean that the null hypothesis is false, but it only means that what you observed (e.g. a specific sequence of coin tosses, if that was the experiment, and, more precisely, the test statistic computed based on what you observed) is "unlikely" if the null hypothesis is true

    - if the p-value >= alpha, you cannot immediately accept the alternative hypothesis, but you can just say that you "fail to reject" the null hypothesis 
    
        - (almost no one really made this conclusion, which I think it's due to the fact that the professor did not emphasize this point/topic); 

        - (however, there's another framework for hypothesis testing where you may accept the alternative hypothesis: see this https://www.sjsu.edu/faculty/gerstman/EpiInfo/pvalue.htm)

    - So, the significance level is also the probability of mistakenly rejecting the null hypothesis (i.e. the probability of rejecting the null hypothesis if it was true);
        - So, the significance level is the probability of making false positives, where
            - "positive" means "to reject the null hypothesis"
            - "false" means "it was incorrect to reject the null hypothesis"

    - So, for example, if significance level alpha = 0.05, it means that there's a 5% probability of rejecting the null hypothesis when it was actually true; so, if p-value is less than 0.05, what we observed is less likely than 5% of occurring under the null hypothesis, but it would still be possible (i.e. the null hypothesis could still be true, although it may also be false)

    - The p-value is highly related to the concept of confidence interval. When we compute e.g. the t-statistic (which is a type of "test statistic", i.e. a statistic used for an hypothesis test) in a t-test, we typically compute it based on the data we observed/collected (e.g. the means of two samples). This t-statistic may follow some distribution (known as the "sampling distribution"). This distribution is what determines what is likely or unlikely for the values of this test statistic. For example, if it's a Gaussian, the things in the middle are more likely (there's more "density" on them). The confidence intervals are the regions of these distributions. The 95% confidence interval means the 95% biggest region of this distribution. So, anything that lies outside e.g. the 95% confidence region is not very likely to be sampled from this distribution. That's why when we obtain a t-statistic outside the 95% confidence interval, we can reject the null hypothesis. The p-value is just the probability associated with the region outside this confidence interval. Remember that the "definite integral" of a probability density function (pdf) is a probability. The p-value is exactly this definite integral of the region outside the confidence interval. This is how p-values and confidence intervals are related. So, you don't need to compute the p-value and at the same time use confidence intervals. You can just use one of them. However, note that there are also confidence values not around the test statistic but around e.g. the sample means. This is slightly different, but they are still related

    - This article https://en.wikipedia.org/wiki/Statistical_significance could be useful
    - This video could also be useful: https://www.youtube.com/watch?v=k1at8VukIbw&ab_channel=jbstatistics
    - https://www.youtube.com/watch?v=3eVUPi25nzo&ab_channel=JohnLevendisJohnLevendis

- The specific hypothesis test you were expected to perform is described in the slides and it should be the one for the regression problem, so not for the classification problem, which is what was done in one of the labs: so if you directly copied from the lab, your solution is not correct

    - Some of you did it correctly, but not all you.

    - You were expected to follow the instructions on slide 17-19 here: https://www.icorsi.ch/mod/resource/view.php?id=633928.

    - Some of you did not pass the correct sequence of values of the sklearn's functions to perform the statistical tests.

## Code/programming (not strictly related to our course)

- Generally, try to group code that does a specific functionality into a function and code that does another thing into another function, then compose functions, so that to avoid repetition. You can also use classes if you are familiar with object-oriented programming (OOP).

- Many of you tried to set the seed for reproducibility, this is a good thing.

- Some of you tried to build the dataset in a loop, but note that NumPy and other ML libraries already come with functions that allow you to concatenate column or row vectors directly in a more efficient way (see vectorized operations in NumPy)

- One of you performed an assignment and thought that would make a copy, but that's not necessarily the case in Python, so be careful.

- One of you made a mistake in creating the new data of features for task 1, so make sure that you read the documentation of the functions so that they do what you're actually expecting. It may be a good idea to double-check the results when you're unsure.


## Early stopping

- Some of you tried to use early stopping to choose a function that has not over-fitted the training data. Unfortunately, many of you did not use the Keras API correctly, as you passed the argument "loss" (rather than "val_loss") to the parameter "monitor". 

- Some of you also split the data intro training, validation and test datasets, and you passed the validation dataset to the Keras' fit method, but this data was not used for early stopping, because, to do that, you need to pass a EarlyStopping object as a callback to fit. This article https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/ describes how to do early stopping in Keras.

## New activation function

- Most of you answered correctly to the question "Would you expect to have better luck with a neural network with activation function `h(x) = - x * e^(-2)` for the hidden units?", but only some of you explained why the new activation function is still linear: it's linear because we're just shifting the x by a constant (i.e. you're just changing the slope of the line), although that constant may look like an exponential function; furthermore, you could also have stated that a neural network, even with multiple hidden layers, but all of them using a linear activation function, would still only be able to solve linear problems, as the composition of linear functions is linear.

## Logistic regression vs perceptron

- Many of you pointed out that one of the differences between logistic regression and the perceptron is the different activation function they use (sigmoid vs step function), but it's also important to note that logistic regression (because of the activation function it uses) produces an output in the range [0, 1], which can be interpreted as a probability, so logistic regression is a probabilistic classifier.

## Logistic regression and linear regression

- Logistic regression is not the same thing as linear regression, but both are instances of a generalised linear model. (Maybe this https://stats.stackexchange.com/q/29325 be useful)

## Accuracy vs MSE

The accuracy is not the same thing as the MSE. The accuracy is defined as the number of correct predictions divided by the total number of predictions, while the MSE is the average (or mean) of the squared errors, where the errors are the differences between the predictions and the correct labels. In the context of regression, the accuracy does not make much sense because most predictions will not be exact (i.e. the output of a regression model can be any real or floating-point number in some range), so the accuracy can be very low or even zero, although the model may still have a quite small MSE. The MSE can be interpreted as a measure of "accuracy" of a regression model, but it's important to understand that "accuracy" has a specific definition in classification problems.

## Complexity of hypothesis class vs complicated function

- Some of you have confused the concept of a hypothesis class (or model family) with a **specific** function (sometimes also called "model") that is chosen from this "model family", and some of you also confused the concept of a complexity (or size) of a model family with a "complicated function" (e.g. a function that is very wiggly, like the [Weierstrass functions](https://en.wikipedia.org/wiki/Weierstrass_function) or some polynomial of very high degree)

    - You can think of the "model family" as a **set** of "models" or functions (in a mathematical sense), so "family" here would be a synonym for "set". 
    
        - So, for example, a **specific** neural network with some **specific set of weights** would represent/compute some **specific** function. If you do not change the architecture of this neural network (e.g. if you do not change the number of hidden layers, the number of neurons in each layer or the activation functions), you will not change the hypothesis class (or model family). If you change the specific values of the weights, though, you will be changing the function that the neural network represents or computes. 

    - While training a neural network with e.g. gradient descent, we are typically changing only the values of these weights and not the architecture (or hypothesis class) of the neural network (although, in principle, it's also possible to change the architecture of the neural network during training); so, during training, we are looking for a function (or model) in the (fixed) set of functions/models. 

    - Some of you thought that, in question 1, the x-axis represented the points in time during training, but it actually represents the **complexity** of the model family; so, in this case, as we change e.g. the number of hidden layers of the neural network (as stated in the assignment), we effectively change the hypothesis class. However, it's important to note that similar plots exist for the case where you observe the training, validation and test errors during training, so while we change the weights (and not) the architecture of the neural network, and that's why some of you may have confused this plot with those ones: in the slides there's also a plot where the x-axis can both be the model complexity or training iterations (in this case, the model complexity refers to how complicated the specific function/model is, e.g. whether it's very wiggly or not, and not the complexity of the hypothesis class, as in the assignment).

        - Another thing that may be confusing is that the word "model" is sometimes used to refer to a "hypothesis class" or "model family" and not to a specific function. The reason is: this function would be a model (or approximation) of the target function (the function we're trying to learn). In the lectures, we also talked about white-box, gray-box and black-box models and all of these may be even more confusing...