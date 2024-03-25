import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.svm import SVC



class SimpleMKL_Multiclass:

    def __init__(self, 
                 kernels, 
                 C,
                 tol,
                 d_min,
                 random_state=0
                ):
        self.kernels = kernels  # List of precomputed kernel matrices
        self.C = C              # Regularization parameter
        self.tol = tol
        self.d = None           # Kernel weights
        self.svm = None         # SVM model
        self.random_state=random_state
        self.d_min=d_min

    def fit(self, X, y):
        M = len(self.kernels)
        N = X.shape[0]
        
        # Initialize kernel weights
        d = np.ones(M) / M
        
        # Define objective function J(d)
        def J(d):
            # Compute combined kernel
            K_combined = sum(d[m] * self.kernels[m] for m in range(M))
            
            # Train SVM and get dual coefficients
            self.svm = SVC(C=self.C, kernel='precomputed',
                           class_weight="balanced",
                           random_state=self.random_state
                          )
            self.svm.fit(K_combined, y)
            
            #number of estimators
            self.num_estimators = len(self.svm.dual_coef_)
            
            J_list = []
            
            for i in range(self.num_estimators):
            
                dual_coef = self.svm.dual_coef_[i]
                support_vectors = self.svm.support_
                
                J_temp = -0.5 * np.dot(dual_coef, np.dot(K_combined[support_vectors][:, support_vectors], dual_coef)) + np.sum(np.abs(dual_coef))
                
                J_list.append(J_temp)
            # Compute objective value
            return np.array(J_list).sum()
        
        
        # Define gradient of J(d)
        def grad_J(d):
            
            grad = np.zeros((self.num_estimators,M))
            
            for m in range(M):
                
                K_m = self.kernels[m]
                
                for i in range(self.num_estimators):
                    
                    dual_coef = self.svm.dual_coef_[i]
                    support_vectors = self.svm.support_
                    grad[i][m] = -0.5 * np.dot(dual_coef, np.dot(K_m[support_vectors][:, support_vectors], dual_coef))
            
            #print(f"grad:{grad}")
            #print(f"grad_sum;{grad.sum(axis=0)}")
            
            return grad.sum(axis=0)

        # Optimization problem to minimize J(d) subject to constraints
        constraints = ({'type': 'eq', 'fun': lambda d: np.sum(d) - 1},  # Sum of weights must be 1
                       {'type': 'ineq', 'fun': lambda d: d})            # Weights must be non-negative
        
        
       
        bounds = [(self.d_min, 1-self.d_min)] * len(self.kernels)
        
        # Solve optimization problem
        opt_res = minimize(J, d, method='SLSQP', jac=grad_J, constraints=constraints,
                           tol=self.tol,bounds=bounds
                          )
        
        
        self.d = opt_res.x  # Store the optimal kernel weights
        
        #print(self.d)

    def predict(self, kernels):
        # Compute combined kernel using the learned weights
        K_combined = sum(self.d[m] * kernels[m] for m in range(len(kernels)))
        return self.svm.predict(K_combined)




class ConstructKernelMatrix:
    
    def __init__(self,
                 first_mod_feature_len, 
                 last_mod_feature_len,
                 kernel1,kernel2,
                 gamma1,gamma2,
                 coef1,coef2,
                 degree1,degree2
        ):
        
        self.first_mod_feature_len = first_mod_feature_len
        self.last_mod_feature_len = last_mod_feature_len
        
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.gamma1=gamma1
        self.gamma2=gamma2
        self.coef1=coef1
        self.coef2=coef2
        self.degree1 = degree1
        self.degree2 = degree2

        
        
    def construct_kernel_matrix(self, X,Z,kernel, gamma, coef, degree):
        
        if kernel =="rbf":
            
            return pairwise_kernels(X,Z, metric=kernel, gamma=gamma)
        
        elif kernel =="poly":
            
            return pairwise_kernels(X,Z, metric=kernel, coef0=coef, degree=degree)
        
        else:
            return pairwise_kernels(X,Z, metric=kernel)
        
        
    def train_kernel_matrix(self, train_data):
        
        
        #slice data for modality 1
        self.data_m1_train = train_data[:,:self.first_mod_feature_len]
        
        #slice data for source/modality 2
        self.data_m2_train = train_data[:,self.first_mod_feature_len:]
        
        
        Kernel_matrix1 = self.construct_kernel_matrix(self.data_m1_train,self.data_m1_train,
                                                      self.kernel1, self.gamma1,
                                                      self.coef1, self.degree1
                                                     )
        
        
        Kernel_matrix2 = self.construct_kernel_matrix(self.data_m2_train,self.data_m2_train,
                                                      self.kernel2, self.gamma2,
                                                      self.coef2, self.degree2
                                                     )
        
        return np.array([Kernel_matrix1,Kernel_matrix2])
    
    
    
    def test_kernel_matrix(self, test_data):
        
        #slice data for modality 1
        data_m1_test = test_data[:,:self.first_mod_feature_len]
        
        #slice data for source/modality 2
        data_m2_test = test_data[:,self.first_mod_feature_len:]
        
        
        Kernel_matrix1 = self.construct_kernel_matrix(data_m1_test,self.data_m1_train,
                                                      self.kernel1, self.gamma1,
                                                      self.coef1, self.degree1
                                                     )
        
        
        Kernel_matrix2 = self.construct_kernel_matrix(data_m2_test,self.data_m2_train,
                                                      self.kernel2, self.gamma2,
                                                      self.coef2, self.degree2
                                                     )
        
        
        return np.array([Kernel_matrix1,Kernel_matrix2])



class Multikernel_SVM(ConstructKernelMatrix,
                      BaseEstimator, ClassifierMixin):
    
    def __init__(self,
                 first_mod_feature_len,
                 last_mod_feature_len,
                 d_min,
                 kernel1,
                 kernel2,
                 gamma1,
                 gamma2,
                 coef1,
                 coef2,
                 degree1,
                 degree2,
                 tol=1e-3,
                 C=1.0,
                 random_state=0   
        ):
        
        self.tol=tol
        self.C=C
        self.first_mod_feature_len=first_mod_feature_len
        self.last_mod_feature_len=last_mod_feature_len
        self.kernel1=kernel1
        self.kernel2=kernel2
        self.gamma1=gamma1
        self.gamma2=gamma2
        self.coef1=coef1
        self.coef2=coef2
        self.degree1=degree1
        self.degree2=degree2
        self.tol=tol
        self.C=C
        self.random_state=random_state  
        self.model = None
        self.d = None
        self.d_min=d_min
         
        super(). __init__(
            first_mod_feature_len=first_mod_feature_len,
            last_mod_feature_len=last_mod_feature_len,
            kernel1=kernel1,
            kernel2=kernel2,
            gamma1=gamma1,
            gamma2=gamma2,
            coef1=coef1,
            coef2=coef2,
            degree1=degree1,
            degree2=degree2,

        )
        
        
    def fit(self, train_data, y_train):
        
        self.classes_ = unique_labels(y_train)
        
        
        self.Kernel_Matrix_Obj = ConstructKernelMatrix(
                                   first_mod_feature_len=self.first_mod_feature_len,
                                   last_mod_feature_len=self.last_mod_feature_len,
                                   kernel1=self.kernel1, 
                                   kernel2= self.kernel2,
                                    gamma1=self.gamma1,
                                    gamma2=self.gamma2,
                                    coef1=self.coef1,
                                    coef2=self.coef2,
                                    degree1=self.degree1,
                                    degree2=self.degree2,
                                  )
        self.train_kernel_matrices = self.Kernel_Matrix_Obj.train_kernel_matrix(train_data)
        
        
        self.model = SimpleMKL_Multiclass(kernels=self.train_kernel_matrices, 
                                        C=self.C,
                                        tol=self.tol,
                                        random_state=self.random_state,
                                        d_min=self.d_min
                                       )
        
        
        self.model.fit(self.train_kernel_matrices, y_train)
        
        self.d = self.model.d
        
        
        return self
        
        
        
    def predict(self, test_data):
        
        self.test_kernel_matrices = self.Kernel_Matrix_Obj.test_kernel_matrix(test_data)
        
        predictions = self.model.predict(self.test_kernel_matrices)
        
        return predictions