import numpy as np
import matplotlib.pyplot as plt
 
def ci_plot(x,y):
 
    # fit a curve to the data using a least squares 1st order polynomial fit
    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    fit = p(x)

    # get the coordinates for the fit curve
    c_y = [np.min(fit),np.max(fit)]
    c_x = [np.min(x),np.max(x)]

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y -p_y

    # create series of new test x-values to predict for
    p_x = np.arange(np.min(x),np.max(x)+1,1)

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)         # mean of x
    n = len(x)              # number of samples in origional fit
    t = 2.31                # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

    confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
                ((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))

    # now predict y based on test x-values
    p_y = z[0]*p_x+z[0]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    # set-up the plot
    # plt.plot(figsize = (8,4))
    # plt.axes().set_aspect('equal')

    plt.plot(figsize = (12,4))
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Linear regression and confidence limits')

    # plot sample data
    plt.plot(x,y,'bo',label='Observations')

    # plot line of best fit
    plt.plot(c_x,c_y,'r-',label='Regression line')

    # plot confidence limits
    plt.plot(p_x,lower,'b--',label='Confidence interval (95%)')
    plt.plot(p_x,upper,'b--')

    # set coordinate limits
    # plt.xlim(0,11)
    # plt.ylim(0,11)

    # configure legend
    plt.legend(loc=0)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)

    # show the plot
    plt.show()