import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.multiclass import OneVsRestClassifier
import copy

class SimpleMKL_Multiclass:
    
    def __init__(self, kernels, beta=0.2,
                 C=10.0, max_iter=100,
                 tol=1e-3, random_state=0):
        
        self.C = C
        self.kernels=kernels
        self.beta=beta #min modality contribution
        self.max_iter = max_iter
        self.tol = tol
        self.random_state=random_state

    def fit(self, X, y):
        
        self.M = len(self.kernels)
        self.d = np.ones(self.M) / self.M

        for iteration in range(self.max_iter):
            K = sum(self.d[m] * self.kernels[m] for m in range(self.M))
            self.svm = OneVsRestClassifier(SVC(C=self.C, 
                                               kernel='precomputed', 
                                               random_state=self.random_state), n_jobs=-1)
            self.svm.fit(K, y)
            obj = self._objective(self.d, self.svm, self.kernels)
            grad = self._compute_gradient(self.svm,self.kernels)
            #print(f"Gradient:{grad}")
            #print(f"support vectors: {self.svm.support_}")
            self.D = self._compute_descent_direction(self.d, grad)
            #print(f"D:{D}")
            #print(f"grad:{grad}")
            #print(f"obj:{obj}")
            obj_= 0
            d_=self.d.copy()
            D_=self.D.copy()
            mu=np.argmax(self.d)
            self.gamma_max=0
          
            while obj_< obj:
                
                d=d_.copy()
                D=D_.copy()
                #print(f"d:{d}")
                #print(f"D:{D}")
                #indices of d element with negative grad
                Dm_negative_index= [m for m in range(len(D)) if D[m]<0] #[2,3]
                #print(f"Negative index Dm:{Dm_negative_index}")
                
                # divide the d elements by their corresponding grad
                d_div_D = ([-d[i]/D[i] for i in Dm_negative_index])  #[-1,-2]
                #print(f"d_div_D:{d_div_D}")

                #determine the index with min value of d_div_D
                try:
                    d_div_D_min = np.argmin(d_div_D)
                
                except:
                    break
                #d_div_D_min = np.argmin(d_div_D)        
                v = Dm_negative_index[d_div_D_min]
                
                gamma_max = -d[v]/D[v]
                #print(f"gamma max:{gamma_max}")
                
                if gamma_max > 0:
                    self.gamma_max = gamma_max
                
                d_ = d + self.gamma_max*D
                D_[mu] = D[mu]-D[v]
                D_[v]=0
                K_ = sum(d_[m] * self.kernels[m] for m in range(self.M))
                svm = OneVsRestClassifier(SVC(C=self.C, kernel='precomputed', 
                                              random_state=self.random_state), n_jobs=-1)
                svm.fit(K_, y)
                obj_ = self._objective(d_, svm, self.kernels)
                #print(f"obj_{obj_}")

            #print(f"gamma max:{self.gamma_max}")
            #print(f"d_value{d}")
            d_updated = self._line_search(self.d, self.D, self.gamma_max, self.svm,y,self.kernels)
            #d=d_updated
            self.d = d_updated
            #print(f"updated d:{d}")
            
    def _objective(self, d, fitted_estimator,kernels):
        
        K = sum(d[m] * kernels[m] for m in range(len(d)))
        
        #number of estimators
        self.num_estimators = len(fitted_estimator.estimators_)
        #print(f"Num of estimators: {self.num_estimators}")

        J_list = []

        for i in range(self.num_estimators):

            dual_coef = fitted_estimator.estimators_[i].dual_coef_
            support_vectors = fitted_estimator.estimators_[i].support_
            J_temp = -0.5 * np.dot(dual_coef, np.dot(K[support_vectors][:,support_vectors], dual_coef.T)) + np.sum(np.abs(dual_coef))
            J_list.append(J_temp)
            
        # Compute objective value
        return np.array(J_list).sum()
        
                                                                                                
    def _line_search(self, d, D, gamma_max,fitted_estimator,y, kernels):
        best_d = d
        clf_copy = copy.deepcopy(fitted_estimator)
        best_obj = self._objective(d,fitted_estimator, kernels)
        
        for gamma in np.linspace(0, gamma_max,15):
            #print(d)
            #print(D)
            d_new = d + gamma * D
            #print(d_new)
            #d_new[d_new < 0] = 0 
            d_new /= np.sum(d_new)
            
            if (d_new <self.beta).sum()>0:
                d_new[d_new<self.beta]=self.beta
                d_new[d_new > self.beta]=1-self.beta
                
            K= sum(d_new[m] * kernels[m] for m in range(self.M))
            clf_copy.fit(K, y)
            obj = self._objective(d_new,clf_copy, kernels)
            if obj < best_obj:
                best_d = d_new
                best_obj = obj
                #print(f"Best d:{best_d}")
        return best_d

    def predict(self, kernels_test):
        K_test = sum([self.d[i] * kernels_test[i] for i in range(len(self.d))])
        return self.svm.predict(K_test)

    def _compute_gradient(self, estimator, kernels):
        #M = len(kernels)
        grad = np.zeros((self.num_estimators,self.M))   
        
        for m in range(self.M):
            K_m = kernels[m]
            
            for i in range(self.num_estimators):
                
                dual_coef = estimator.estimators_[i].dual_coef_
                support_vectors = estimator.estimators_[i].support_
                grad[i][m] = -0.5 * np.dot(dual_coef, np.dot(K_m[support_vectors][:, support_vectors], dual_coef.T))
        #print(grad)  
        return grad.sum(axis=0)
    
    def _compute_descent_direction(self, d, grad):
        d_ = d.copy()
        #d_[d_<self.beta]=self.beta
        mu = np.argmax(d_)
        D = np.zeros_like(d_)
        
        for m in range(len(d_)):
            #if d[m] == 0 and (grad[m] - grad[mu]) > 0:
            if d_[m] == self.beta and (grad[m] - grad[mu]) > 0:
                D[m] = 0
            #elif d[m] > self.beta and m!= mu:
            elif d_[m] > self.beta and m!= mu:
                D[m] = -grad[m] + grad[mu]
            elif m==mu:
                D[m] = np.sum([grad[nu] - grad[mu] for nu in range(len(d_)) if nu != mu and d_[nu] > self.beta])
        return D