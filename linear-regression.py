import numpy as np
from sklearn.datasets.samples_generator import make_regression
import sklearn.linear_model as lm
import math


# the version of gradient descent which computes theta values for multi value_linear_regression
def gradient_descent_multiple(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples

    # initial theta
    t0 = np.random.random()
    t1 = np.random.random()
    t2 = np.random.random()
    t3 = np.random.random()

    # total error, J(theta)
    k = 0
    J = sum([(t0 + t1 * x[i][k] + t2 * x[i][k+1] + t3 * x[i][k+2] - y[i]) ** 2 for i in range(m)])

    p = 1
    # Iterate Loop
    while not converged:


        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0 / m * sum(
            [(t0 + t1 * x[i][k] + t2 * x[i][k+1] + t3 * x[i][k+2] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum(
            [(t0 + t1 * x[i][k] + t2 * x[i][k+1] + t3 * x[i][k+2] - y[i]) * x[i][k] for i in range(m)])
        grad2 = 1.0 / m * sum(
            [(t0 + t1 * x[i][k] + t2 * x[i][k + 1] + t3 * x[i][k + 2] - y[i]) * x[i][k+1] for i in range(m)])
        grad3 = 1.0 / m * sum(
            [(t0 + t1 * x[i][k] + t2 * x[i][k + 1] + t3 * x[i][k + 2] - y[i]) * x[i][k+2] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
        temp2 = t2 - alpha * grad2
        temp3 = t3 - alpha * grad3

        # update theta
        t0 = temp0
        t1 = temp1
        t2 = temp2
        t3 = temp3

        y_predict = []

        for i in range(x.shape[0]):
            y_predict.append(t0 + t1 * x[i][k] + t2 * x[i][k + 1] + t3 * x[i][k + 2])

        p=p+1

        # mean squared error
        e = sum([(t0 + t1 * x[i][k] + t2 * x[i][k+1] + t3 * x[i][k+2] - y[i]) ** 2 for i in range(m)])

        if abs(J - e) <= ep:
            print ('Converged, iterations: ', iter, '!!!')
            converged = True

        J = e  # update error
        iter += 1  # update iter

        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0, t1, t2, t3


# the version of gradient descent which computes theta values for single value_linear_regression
def gradient_descent_single(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples

    # initial theta
    t0 = np.random.random()
    t1 = np.random.random()

    # total error, J(theta)
    J = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

    p = 1
    # Iterate Loop
    while not converged:


        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        p=p+1

        # mean squared error
        e = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

        if abs(J - e) <= ep:
            print ('Converged, iterations: ', iter, '!!!')
            converged = True

        J = e  # update error
        iter += 1  # update iter

        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0, t1


# the version of gradient descent which computes theta values for weighted single value_linear_regression
def gradient_descent_weighted(alpha, x, y, ep, max_iter, qp):

    converged = False
    iter = 0

    # sort x data list and add in x_sorted then add x_sorted data's within band width (e.g = 10) range
    # to x_bw so the same for y_bw
    # temp dictionary is for matching x and y when they get sorted
    temp = {}
    for i in range(100):
        temp.update({x[i]: y[i]})

    y_sorted = []

    x_sorted = sorted(x[0:100])
    for i in range(100):
        y_sorted.append(temp[x_sorted[i]])

    bw = min(range(len(x_sorted)), key=lambda i: abs(x_sorted[i] - qp))
    x_bw = x_sorted[bw - 10:bw] + x_sorted[bw:bw + 10]
    y_bw = y_sorted[bw - 10:bw] + y_sorted[bw:bw + 10]

    m = len(x_bw)  # number of samples

    #calculating weight for all samples from query point
    taw = 0.08
    weight_list = []

    for i in range(m):
        weight_list.append( math.exp(-((x_bw[i]-qp) ** 2 / 2 * (taw ** 2))) )

    # initial theta
    t0 = np.random.random()
    t1 = np.random.random()

    # total error, J(theta)
    J = sum([ (weight_list[i] * (t0 + t1 * x_bw[i] - y_bw[i])) ** 2 for i in range(m)])

    p = 1
    # Iterate Loop
    while not converged:


        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0 / m * sum([weight_list[i] * (t0 + t1 * x_bw[i] - y_bw[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([weight_list[i] * (t0 + t1 * x_bw[i] - y_bw[i]) * x_bw[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        p=p+1

        # mean squared error
        e = sum([ (weight_list[i] * (t0 + t1 * x_bw[i] - y_bw[i])) ** 2 for i in range(m)])

        if abs(J - e) <= ep:
            print ('Converged, iterations: ', iter, '!!!')
            converged = True

        J = e  # update error
        iter += 1  # update iter

        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0, t1


# the function generating hypothesis value for given theta's and x's in multi value linear regression
def predicted_multiple(t0, t1, t2, t3, x1, x2, x3):
    return t0 + t1 * x1 + t2 * x2 + t3 * x3

# the function generating hypothesis value for given theta's and x's in single value linear regression
def predicted_single(t0, t1, x):
    return t0 + t1 * x

# the function generating MSE value for given theta's and x's and y's in multi value linear regression
def generalization_error(t0, t1, t2, t3, x, y):
    k = 0
    for i in range(len(x)):
        y_predict = t0 + t1 * x[i][k] + t2 * x[i][k + 1] + t3 * x[i][k + 2]
        result = (y_predict - y[i]) ** 2
        print("The generalization error for sample ", x[i]," is = ", result)

# the library function for generating linear regression model based on given x , y
# and computing the theta values
def multiple_linear_regression(x, y):
    linear_regression = lm.LinearRegression()
    return linear_regression.fit(x, y)


if __name__ == '__main__':

    print("This program is about : \n Linear Regression and locally weighted linear regression ")
    print("Enter your method of generating linear regression problem data")
    choice = int(input("1- Use D3.csv file \t 2- Use library function make_regression = "))

    if choice == 1:
        def read_file(file_name):
            return np.genfromtxt(file_name, delimiter=',')

        data = read_file('D3.csv')

        x = data[:, 0:3]
        y = data[:, 3:4]

    else:
        x, y = make_regression(n_samples=100, n_features=3, n_informative=3,random_state=0, noise=35)


    print ('x.shape = %s y.shape = %s' % (x.shape, y.shape))

    if choice == 1:
        alpha = 0.03  # learning rate
    else:
        alpha = 0.1

    ep = 0.01  # convergence criteria

    # call gradient decent, and get theta0, theta1, theta2, theta3
    theta0, theta1, theta2, theta3 = gradient_descent_multiple(alpha, x, y, ep, max_iter=1000)
    print('theta0 = %s theta1 = %s theta2 = %s theta3 = %s' % (theta0, theta1, theta2, theta3))

    # part 1- a)
    print("\n Answer ---------------- \n 1- a) ")
    linear_model = str(theta0) + " + " + str(theta1) + " x1" + " + " + str(theta2) + " x2" + " + " + str(theta3) + " x3"
    print("The linear model I found = ", linear_model)
    print("The predicted value for set of (1,1,1) is = ",
          predicted_multiple(theta0, theta1, theta2, theta3, 1, 1, 1))

    print("The predicted value for set of (2,0,4) is = ",
          predicted_multiple(theta0, theta1, theta2, theta3, 2, 0, 4))

    print("The predicted value for set of (3,2,1) is = ",
          predicted_multiple(theta0, theta1, theta2, theta3, 1, 1, 1))

    # part 1-b)
    print("\n Answer ---------------- \n 1- b) ")
    print("For predicting generalization error using cross validation we shrink the data set",
          "\n to A training set and test set with variables of x_train and x_test and using",
          "\n gradient_descent to generate new theta's and compare the predict with x_test \n")

    x_train = x[0:95]
    y_train = y[0:95]
    x_test = x[95:100]
    y_test = y[95:100]

    new_theta0, new_theta1, new_theta2, new_theta3 = gradient_descent_multiple(alpha, x_train, y_train, ep, max_iter=1000)
    print('theta0 = %s theta1 = %s theta2 = %s theta3 = %s' % (new_theta0, new_theta1, new_theta2, new_theta3))
    generalization_error(new_theta0, new_theta1, new_theta2, new_theta3, x_test, y_test)

    # part 1-c)
    print("\n Answer ---------------- \n 1- c) ")

    #the coef and intercepts which are the theta values generated by library function sklearn.linear_model
    # for comparing the GD hand written algorithm accuracy to library function
    model = multiple_linear_regression(x, y)
    intercept = model.intercept_

    if choice == 1 :
        slopes = model.coef_[0]
    else:
        slopes = model.coef_

    print("The comparison of theta's from mine version of GD and linear model library")
    print("GD theta0 = ", theta0," library theta0 = ", intercept)
    print("GD theta1 = ", theta1, " library theta1 = ", slopes[0])
    print("GD theta2 = ", theta2, " library theta2 = ", slopes[1])
    print("GD theta3 = ", theta3, " library theta3 = ", slopes[2])

    #part 2-a)
    print("\n Answer ---------------- \n 2- a) ")
    x_new = x[:, 0]
    y_new = y
    t0, t1 = gradient_descent_single(alpha, x_new, y_new, ep, max_iter=1000)
    print("The linear model I found = ", str(t0) + " + " + str(t1) + " x1")
    print("The predicted value for x = 0.5 is = ", predicted_single(t0, t1, 0.5))
    print("The predicted value for x = 1 is = ", predicted_single(t0, t1, 1))
    print("The predicted value for x = 1.5 is = ", predicted_single(t0, t1, 1.5))
    print("The predicted value for x = 2 is = ", predicted_single(t0, t1, 2))
    print("The predicted value for x = 2.5 is = ", predicted_single(t0, t1, 2.5))
    print("The predicted value for x = 3 is = ", predicted_single(t0, t1, 3))

    # part 2-b) and 2-c)
    # computing theta values for each query point (e.g 0.5 - 1 - 1.5) separately using
    # gradient_descent_weighted( ) function and calculating prediction value using predicted_single( )
    print("\n Answer ---------------- \n 2- b) ")
    t0_05 , t1_05 = gradient_descent_weighted(alpha, x_new, y_new, ep, max_iter=10000, qp=0.5)
    print("1- The simple linear regression prediction for 0.5 = ", predicted_single(t0, t1, 0.5))
    print("\t the weighted linear regression prediction for 0.5 = ", predicted_single(t0_05 , t1_05, 0.5))
    print("******")
    t0_1, t1_1 = gradient_descent_weighted(alpha, x_new, y_new, ep, max_iter=10000, qp=1)
    print("2- The simple linear regression prediction for 1 = ", predicted_single(t0, t1, 1))
    print("\t the weighted linear regression prediction for 1 = ", predicted_single(t0_1, t1_1, 1))
    print("******")
    t0_15, t1_15 = gradient_descent_weighted(alpha, x_new, y_new, ep, max_iter=10000, qp=1.5)
    print("3- The simple linear regression prediction for 1.5 = ", predicted_single(t0, t1, 1.5))
    print("\t the weighted linear regression prediction for 1.5 = ", predicted_single(t0_15, t1_15, 0.5))
    print("******")
    t0_2, t1_2 = gradient_descent_weighted(alpha, x_new, y_new, ep, max_iter=10000, qp=2)
    print("1- The simple linear regression prediction for 2 = ", predicted_single(t0, t1, 2))
    print("\t the weighted linear regression prediction for 2 = ", predicted_single(t0_2, t1_2, 2))
    print("******")
    t0_25, t1_25 = gradient_descent_weighted(alpha, x_new, y_new, ep, max_iter=10000, qp=2.5)
    print("1- The simple linear regression prediction for 2.5 = ", predicted_single(t0, t1, 2.5))
    print("\t the weighted linear regression prediction for 2.5 = ", predicted_single(t0_25, t1_25, 2.5))
    print("******")
    t0_3, t1_3 = gradient_descent_weighted(alpha, x_new, y_new, ep, max_iter=10000, qp=3)
    print("1- The simple linear regression prediction for 3 = ", predicted_single(t0, t1, 3))
    print("\t the weighted linear regression prediction for 3 = ", predicted_single(t0_3, t1_3, 3))
    print("******")
