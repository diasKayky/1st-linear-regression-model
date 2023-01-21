import statistics as stats
import numpy as np


class LinearRegression:

    def find_coef(self, x, y):

        # Calculates the slope of the curve

        self.x = x
        self.y = y

        if type(x) == 'pandas.core.frame.DataFrame':
            hh = np.array(x)
            t = np.ndarray.tolist(hh)

        if type(x) == 'numpy.ndarray':
            t = np.ndarray.tolist(x)

        np_x = np.array(x)
        np_y = np.array(y)
        cov = np.cov(np_x, np_y)[0][1]

        return cov / stats.variance(x)

    def intercept(self, x, y):

        # Calculates the intercept of the curve
        self.x = x
        self.y = y

        if type(x) == 'pandas.core.frame.DataFrame':
            hh = np.array(x)
            t = np.ndarray.tolist(hh)

        if type(x) == 'numpy.ndarray':
            t = np.ndarray.tolist(x)

        np_x = np.array(x)
        np_y = np.array(y)

        cov = np.cov(np_x, np_y)[0][1]
        coef = cov / stats.variance(x)

        mean_x = stats.mean(x)
        mean_y = stats.mean(y)
        take = mean_y - (coef * mean_x)

        return round(take, 3)

    def predict(self, x, y):

        # Based on the linear regression model, it predicts the dependent variable from given independent variable data

        self.x = x
        self.y = y

        if type(x) == 'pandas.core.frame.DataFrame':
            hh = np.array(x)
            t = np.ndarray.tolist(hh)

        if type(x) == 'numpy.ndarray':
            t = np.ndarray.tolist(x)

        np_x = np.array(x)
        np_y = np.array(y)

        cov = np.cov(np_x, np_y)[0][1]
        coef = cov / stats.variance(x)

        mean_x = stats.mean(x)
        mean_y = stats.mean(y)

        intercept = mean_y - (coef * mean_x)

        result = []

        for i in x:
            result.append((coef * i) + intercept)

        if type(x) == 'pandas.core.frame.DataFrame' or 'numpy.ndarray':
            hh = np.array(result)

        return result

    def r_sq(self, x, y):

        # Calculates R^2 value for given data

        self.x = x
        self.y = y

        if type(x) == 'pandas.core.frame.DataFrame':
            hh = np.array(x)
            t = np.ndarray.tolist(hh)

        if type(x) == 'numpy.ndarray':
            t = np.ndarray.tolist(x)

        np_x = np.array(x)
        np_y = np.array(y)

        corr = np.corrcoef(np_x, np_y)[0][1]

        return corr ** 2
