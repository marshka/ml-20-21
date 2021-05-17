from tensorflow.keras.models import load_model

if __name__ == '__main__':

    # Load the data

    # ...


    # Preprocessing

    # ...


    # Load the trained models
    #for example
    model_task2 = load_model('./nn_task2.h5')


    # Predict on the given samples
    #for example
    y_pred_task2 = model_task2.predict(x_test)

    # Evaluate the missclassification error on the test set
    # for example
    assert y_test.shape == y_pred_task2.shape
    acc = ...  # evaluate accuracy with proper function
    print("Accuracy model task 2:", acc)
