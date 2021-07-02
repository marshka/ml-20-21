# General feedback for assignment 2

## ML concepts

- **Early stopping with the correct metric**: Some of you, as in the first assignment, did not use the `EarlyStopping` class correctly. Specifically, some of you monitored the training accuracy (or even loss) by passing the `"accuracy"` (or `"loss"`, respectively) to the `monitor` parameter of the class `EarlyStopping`, but you should have passed `"val_accuracy"`. This is an big mistake, so make sure not to make it again in the future. 

    - It's also important to note that, in general, the metrics that you may want to monitor (like the accuracy) may not correspond to the loss. In the case of a multi-class classification problem (like exercise 1 of this assignment 2), the loss was the cross-entropy and the metric was the accuracy, which are different! 

- **Correct loss function**: one of you tried to use the _**sparse** categorical cross-entropy_ from _logits_, but given that we used the _softmax_ activation function for the output layer, the output layer produces a probability distribution and not "logits" (which would be the non-normalized outputs of the output layer, i.e. if you did not use the softmax but a linear/no activation function)

- **Validation vs testing**: You should not have used the test data for validation. You should have either manually split the training data into training and validation data or specify the `0.2` as the value for the parameter `validation_split` of the `fit` method (and, in both cases, you should not have touched the test data, which is used for model evaluation, i.e. assess the generalisation ability of the model). This is a big mistake because cross-validation is extremely important in machine learning and you really need to understand it.

- **Transfer learning**: Some of you did not really performed transfer learning in task 2. It was really required to that you used VGG16 as the backbone of the neural network. I don't know whether you didn't do this because of lack of time, computational resources (to train the bigger model) or you misunderstood the specifications of the assignment.

- **Analyse the plots**: Nobody commented on why, in the second task, in most cases, with data augmentation, the validation accuracy was higher than the training accuracy. Isn't that strange? Did you also augment the validation data? Would this change if we trained the model for more epochs?

## Programming mistakes and style

- **Tuples and function calls**: In Python, `(2, 2)` would represent a tuple of two numbers. If the first parameter of a function `f` requires a tuple and you call that function as follows `f(1, 2)`, then you will **not** be passing a tuple to the first parameter, but just the number `1`. You need to do this `f((1, 1), 2)`.

- **4 spaces**: In Python, although this is just a (more or less standard) convention, you should use 4 spaces for the indentation. In any case, the most important thing is to be consistent, i.e. if you use 2 spaces for the indentation of a function, you should use 2 spaces for the indentation of all other functions, because, otherwise, this can lead to some problems/bugs. See https://www.python.org/dev/peps/pep-0008/#indentation for more details.

- **Execution of scripts**: You don't need to define the functions inside the `if __name__ == "__main__"` for them to be visible inside that block. So, you can define them outside that `if __name__ == "__main__"` block, and then call them from inside the block. The `if __name__ == "__main__"` is used to define the block of code that is executed when you run that Python file directly, but this Python file (or module) could still be imported from another Python file (for example, in the case you need to use some function from that module); take a look at https://stackoverflow.com/q/419163/3924118 or https://docs.python.org/3/library/__main__.html for more details.

- **Avoid repetitions (and use functions and/or classes)**: In many cases, some of you rewrote the same code again and again. When that happens, it's a good indication that you could put that code into a function, which is called whenever you need that piece of code. For example, in some cases, you wrote the same code more than once to define the model object. You could have just defined a function that creates the model object, and call that function when you need a new object of that `Sequential` class (for instance). In other cases, you just copied and pasted the code from another module rather than importing that code from that module.

- **Code readability (with spaces)**: In some cases, some of you did not use any spaces between the definition of a function and other functions or other code. To improve the readability of your code, I would say that you should leave at least one space between logically distinct pieces of code (e.g. leave a space between the definition of a function and the call to the same function). See https://www.python.org/dev/peps/pep-0008/#blank-lines for more details. Some IDEs (such as [PyCharm](https://www.jetbrains.com/pycharm/guide/tips/reformat-code/) or IntelliJ have some commands/shortcuts that allow you to format the code so that it's compliant with the [PEP8](https://www.python.org/dev/peps/pep-0008/) conventions)

- **Avoid repetition and use correct programming  abstractions**: In case you need to do something where you just need to change some parameters or hyper-parameters but everything else is the same  (e.g. grid search), it may be a good idea to use loops and iterate through the possible values of the hyper-parameters (rather than hardcoding everything).

- **Comments/conditions**: One of you commented the code for training to avoid training the model again. Rather than commenting, you can use `if` conditions whose corresponding blocks (e.g. with the training code) are executed when the conditions are true. The condition in this case would be something like `training`, which would be set to `False` when you don't want to train the model, and `True` otherwise.

- **Reproducibility of the experiments** (seed + [example](https://stackoverflow.com/help/minimal-reproducible-example)): in ML, when performing an experiment, it's very important to implement it in such a way that later is reproducible. One way to do that is to set the seed of the random number generations. Another way is to provide the code that we can run and check that the reported results correspond to the result of the execution of that code.

