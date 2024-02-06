import numpy as np
import pandas as pd
import EUSCHC as eus
import imp
imp.reload(eus)
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state, check_X_y, check_array


class EUSBoost(AdaBoostClassifier):

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=None,
        n_population=30, 
        n_generation=50 
    ):

        self.algorithm = algorithm
        self.n_population = n_population
        self.n_generation = n_generation
        self.eus = eus.EUS_CHC(n_population=self.n_population, n_generation=self.n_generation)
        self.best_chromosomes = []  

        super(EUSBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def evoundersample(self, X, y, iboost):
        pos_size = len((y==1).nonzero()[0])
        neg_size = len((y==0).nonzero()[0])
        pos_data = X[y==1]
        neg_data = X[y==0]
        
        if pos_size > neg_size:
            self.major_data = pos_data
            self.y_major = y[y==1]
            self.minor_data = neg_data
            self.y_minor = y[y==0]
            self.minor = 0
        else:
            self.minor_data = pos_data
            self.y_minor = y[y==1]
            self.major_data = neg_data
            self.y_major = y[y==0]
            self.minor = 1
        
        df_maj = pd.concat([pd.DataFrame(self.major_data), pd.Series(self.y_major)], axis=1)
        df_min = pd.concat([pd.DataFrame(self.minor_data), pd.Series(self.y_minor)], axis=1)

        X_best= self.eus.under_sampling(df_maj, df_min, iboost, self.best_chromosomes)

        self.best_chromosomes.append(X_best)              

        minor_idx = (y == self.minor).nonzero()[0]
        major_idx = (y == int(not self.minor)).nonzero()[0]
        major_idx = major_idx[np.where(np.asarray(X_best, dtype=object) == 1)]
        print("Major Index Len:",len(major_idx))
        print("Minor Index Len:",len(minor_idx))
        
        return sorted(np.concatenate((minor_idx, major_idx)))


    def fit(self, X, y, sample_weight=None):
        # Check that algorithm is supported.
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or isinstance(
            self.base_estimator, (BaseDecisionTree, BaseForest)
        )):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            dtype = None
            accept_sparse = ["csr", "csc"]

        X, y = check_X_y(
            X,
            y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            y_numeric=is_regressor(self),
        )

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples."
                )

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)

        for iboost in range(self.n_estimators):
            print("Looping estimator ke-",iboost)
            # Evolutionary Under Sampling step
            eus_idx = self.evoundersample(X, y, iboost)

            # Boosting step.
            sample_weight[eus_idx], estimator_weight, estimator_error = self._boost(
                iboost,
                X[eus_idx],
                y[eus_idx],
                sample_weight[eus_idx],
                random_state,
            )
            

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
    