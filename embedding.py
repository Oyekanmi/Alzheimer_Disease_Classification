import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class EMR_SLRA:
    
    def __init__(self,
                d,
                lambda1,lambda2,
                n_neighbors,r,Iter,\
                eps,t,first_mod_dim,
                seed=0
                ):
        self.d = d
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_neighbors=n_neighbors
        self.r=r
        self.Iter=Iter
        self.eps=eps
        self.t = t
        self.first_mod_dim = first_mod_dim
        #self.last_mod_dim=last_mod_dim
        self.seed = seed


    def compute_similarity_matrix(self, X, n_neighbors, t):
    # X is the feature matrix for the v-th view with shape (N, M)
    # k is the number of nearest neighbors
    # t is the parameter in the Gaussian function

        N = X.shape[0]  # Number of samples
        S = np.zeros((N, N))  # Initialize the weight matrix with zeros

        # Use NearestNeighbors from sklearn to find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Compute the weight matrix
        for i in range(N):
            for j in indices[i]:
                if i != j :  # Exclude self-loops
                    S[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / t)
                    S[j, i] = S[i, j]  # Ensure the matrix is symmetric

        return S


    #caculate the laplacian matrix
    def calculate_Laplacian(self, X, n_neighbors, t, \
                            first_mod_dim
                        ):
        
        mri_features = X[:,:first_mod_dim]
        pet_features = X[:,first_mod_dim:]
        
        
        W_MRI = self.compute_similarity_matrix(mri_features, n_neighbors, t)
        WMRI_sum = np.sum(W_MRI, axis=0)
        D1= np.diag(WMRI_sum)
        L1 = D1-W_MRI
        
        W_PET = self.compute_similarity_matrix(pet_features, n_neighbors, t)
        WPET_sum = np.sum(W_PET, axis=0)
        D2= np.diag(WPET_sum)
        L2 = D2-W_PET
        L = np.array([L1,L2])

        return L

    def calculate_Obj(self, X_hat, X, Y,U,L,V, beta, lambda1, lambda2):
    
        term1 = np.linalg.norm(X_hat - U @ Y, 'fro')**2
        term2=lambda1 * sum(beta[v]**r * np.trace(Y @ L[v] @ Y.T) for v in range(V))
        term3 = lambda2* np.linalg.norm(X_hat- X, ord=2,axis=1).sum()
        
        obj = term1+term2+term3
        
        return obj

    def calculate_D_matrix(self, X_hat, X):
    
        frobenius_norms = np.linalg.norm(X_hat - X, axis=1)
        D_matrix = np.diag(1 / (2 * frobenius_norms))
        
        return D_matrix

    def joint_feature_embedding(self,X_arr):
                
        
        #calculate the Laplacian matrix 
        L = self.calculate_Laplacian(X_arr, self.n_neighbors, 
                                    self.t,self.first_mod_dim
                                    )
        X = X_arr.T
        m, N = X.shape
        V = 2 #len(X)  #number of modalities ; MRI & PET
        #print(f"X shape: {X.shape}")
        #calculate the laplacian
        
        #print(f"L:{L.shape}")
        # Initialization
        np.random.seed(self.seed)
        Yt = np.random.rand(self.d, N)
        X_hat = X + np.random.rand(m, N)
        beta = np.array([1/V]*V)
        #print(f"Initial Beta: {beta}")
        D_matrix = self.calculate_D_matrix(X_hat, X) 
        #print(f"D: {D.shape}")
        Objt=0
        min_obj = np.inf
        result = {}
        
        for t in range(self.Iter):
            # SVD decomposition
            #print(f"Iter :{t+1}")
            #print(f"\n")
            #print(f"Yt: {Yt.shape}")
            #print(f"Xhat: {X_hat.shape}")
            Zt = X_hat @ Yt.T  # m x N x N x d == > m x d
            print(f"Zt dim: {Zt.shape}")
            
            try:
                #print(Zt.shape)
                Gt, Dt, Vt = scipy.linalg.svd(Zt,full_matrices=False)
            except:
                print(f"No convergence for SVD....Skip")
                continue
            # Update Ut+1
            #print(f"Gt: {Gt.shape}") # m x d
            #print(Gt)
            #print(f"Vt: {Vt.shape}") # d x d
            #print(f"Dt: {Dt.shape}") # d x d
            #print(Dt)
            Ut= Gt @ Vt.T #m x d
            #print(f"Ut: {Ut.shape}") 
            
            # Update Yt+1
            Lt = sum([(beta[v]**self.r) * L[v] for v in range(V)])
            #print(f"Lt shape: {Lt.shape}")
            Psi_t = np.eye(N) + self.lambda1 * Lt # N x N
            #print(f"Psi_t shape: {Psi_t.shape}")
            #Yt_next = Ut_next.T @ np.linalg.inv(Psi_t) @ X_hat
            Yt_next = Ut.T @ X_hat @ np.linalg.inv(Psi_t)  # N x N # it should be Ut.T instead of Ut_next.T
            #print(f"Yt_next:{Yt_next.shape}")
            
            # Update beta
            p_t = np.array([np.trace(Yt@ L[v] @ Yt.T) for v in range(V)]) #
            #print(f"p_t shape: {p_t.shape}")
            #beta = (r * p_t)**(1 / (1 - r)) / sum((r * p_t)**(1 / (1 - r)))

            beta = [(self.r * p_t[v])**(1 / (1 - self.r)) for v in range(V)]
            beta /= sum(beta)
            #print(f"beta shape: {beta.shape}")
            #print(f"Iter{t+1}: beta:{beta}")
            
            
            # Update X_hat and W_hat
            #X_hat_next = np.linalg.inv(np.eye(d) + lambda2 * D) @ (Ut_next @ Yt_next + lambda2 * D @ X)
            #print(f"W_hat: {W_hat.shape}")
            #(Ut @ Yt)==> 68 x194
            a=np.linalg.inv(np.eye(m) + self.lambda2 * D_matrix)
            b=(Ut @ Yt) + (self.lambda2 *D_matrix@X)
            #print(f"a:{a.shape}")
            #print(f"b:{b.shape}")
            X_hat_next = a @ b
       
            #print(f"X_hat_next  shape: {X_hat_next.shape}")
            
            D_matrix_next = self.calculate_D_matrix(X_hat_next, X)
            
            # Calculate Objt+1 and check for convergence
            Objt_next = self.calculate_Obj(X_hat_next,X, Yt_next,Ut,L,V,beta, self.lambda1, self.lambda2) 
            
            if Objt_next < min_obj:
                #print("Best Minimum.....Storing results")
                result["loss"]=Objt_next
                result["Yt"]=Yt_next
                result["Ut"]=Ut
                result["iter"]=t+1
                min_obj=Objt_next
            
            #print(Objt_next)
            
            if abs(Objt_next - Objt) < self.eps:
                break
            
            # Update variables for the next iteration
            Yt, X_hat, D_matrix, Objt = Yt_next, X_hat_next, D_matrix_next, Objt_next
            #print("\n")
            #print("\n")
        
        return result