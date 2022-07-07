import pandas as pd 
from fim import fpgrowth,fim # you can comment this out if you do not use fpgrowth to generate rules
import numpy as np
from itertools import chain, combinations
import itertools
from numpy.random import random
from scipy import sparse
from bisect import bisect_left
from random import sample
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier

class hyb(object):
    def __init__(self, binary_data,Y,Yb, alpha = 1, beta = 0.1):
        self.df = binary_data  
        self.Y = Y
        self.N = float(len(Y))
        self.Yb = Yb
        self.alpha = alpha
        self.beta = beta 
          

    def generate_rulespace(self,supp,maxlen,N, need_negcode = False,method = 'fpgrowth'):
        if method == 'fpgrowth': # generate rules with fpgrowth
            if need_negcode:
                df = 1-self.df 
                df.columns = [name.strip() + 'neg' for name in self.df.columns]
                df = pd.concat([self.df,df],axis = 1)
            else:
                df = 1 - self.df
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
            itemMatrix = [[item for item in df.columns if row[item] ==1] for i,row in df.iterrows() ]  
            prules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            prules = [np.sort(x[0]).tolist() for x in prules]
            nrules= fpgrowth([itemMatrix[i] for i in nindex],supp = supp,zmin = 1,zmax = maxlen)
            nrules = [np.sort(x[0]).tolist() for x in nrules]
        else: # if you cannot install the package fim, then use random forest to generate rules
            print('Using random forest to generate rules ...')
            prules = []
            for length in range(2,maxlen+1,1):
                n_estimators = 250*length
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,self.Y)
                for n in range(n_estimators):
                    prules.extend(extract_rules(clf.estimators_[n],self.df.columns))
            prules = [list(x) for x in set(tuple(np.sort(x)) for x in prules)] 
            nrules = []
            for length in range(2,maxlen+1,1):
                n_estimators = 250*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df,1-self.Y)
                for n in range(n_estimators):
                    nrules.extend(extract_rules(clf.estimators_[n],self.df.columns))
            nrules = [list(x) for x in set(tuple(np.sort(x)) for x in nrules)]   
            df = 1-self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df,df],axis = 1)
        self.prules, self.pRMatrix, self.psupp, self.pprecision, self.perror = self.screen_rules(prules,df,self.Y,N,supp)
        self.nrules, self.nRMatrix, self.nsupp, self.nprecision, self.nerror = self.screen_rules(nrules,df,1-self.Y,N,supp)

    def screen_rules(self,rules,df,y,N,supp,criteria = 'precision'):
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        # mat = sparse.csr_matrix.dot(df,ruleMatrix)
        mat = np.matrix(df)*ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)

        Zpos = [Z[i] for i in np.where(y>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        supp_select = np.where(TP>=supp*sum(y)/100)[0]
        FP = np.array(np.sum(Z,axis = 0))[0] - TP
        precision = TP.astype(float)/(TP+FP)


        supp_select = np.array([i for i in supp_select if precision[i]>np.mean(y)])
        select = np.argsort(precision[supp_select])[::-1][:N].tolist()
        ind = list(supp_select[select])
        rules = [rules[i] for i in ind]
        RMatrix = np.array(Z[:,ind])
        rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in rules]
        supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind]
        return rules, RMatrix, supp, precision[ind], FP[ind]


    def train(self, Niteration = 5000, print_message=False, interpretability = 'size'):
        self.maps = []
        int_flag = int(interpretability =='size')
        T0 = 0.01
        nprules = len(self.prules)
        pnrules = len(self.nrules)
        prs_curr = sample(list(range(nprules)),3)
        nrs_curr = sample(list(range(pnrules)),3)
        obj_curr = 1000000000
        obj_min = obj_curr
        self.maps.append([-1,obj_curr,prs_curr,nrs_curr,[]])
        p = np.sum(self.pRMatrix[:,prs_curr],axis = 1)>0
        n = np.sum(self.nRMatrix[:,nrs_curr],axis = 1)>0
        overlap_curr = np.multiply(p,n)
        pcovered_curr = p ^ overlap_curr
        ncovered_curr = n ^ overlap_curr
        covered_curr = np.logical_xor(p,n)
        Yhat_curr,TP,FP,TN,FN  = self.compute_obj(pcovered_curr,covered_curr)
        nfeatures = len(np.unique([con.split('_')[0] for i in prs_curr for con in self.prules[i]])) + len(np.unique([con.split('_')[0] for i in nrs_curr for con in self.nrules[i]]))
        obj_curr = ( FN + FP)/self.N +self.alpha*(int_flag *(len(prs_curr) + len(nrs_curr))+(1-int_flag)*nfeatures)+ self.beta * sum(~covered_curr)/self.N
        self.actions = []
        for iter in range(Niteration):
            if iter >0.75 * Niteration:
                prs_curr,nrs_curr,pcovered_curr,ncovered_curr,overlap_curr,covered_curr, Yhat_curr = prs_opt[:],nrs_opt[:],pcovered_opt[:],ncovered_opt[:],overlap_opt[:],covered_opt[:], Yhat_opt[:] 
            prs_new,nrs_new , pcovered_new,ncovered_new,overlap_new,covered_new= self.propose_rs(prs_curr[:],nrs_curr[:],pcovered_curr[:],ncovered_curr[:],overlap_curr[:],covered_curr[:], Yhat_curr[:], obj_min,print_message)
            self.Yhat_curr = Yhat_curr
            Yhat_new,TP,FP,TN,FN = self.compute_obj(pcovered_new,covered_new)
            self.Yhat_new = Yhat_new
            nfeatures = len(np.unique([con.split('_')[0] for i in prs_new for con in self.prules[i]])) + len(np.unique([con.split('_')[0] for i in nrs_new for con in self.nrules[i]]))
            obj_new = (FP + FN)/self.N +self.alpha*(int_flag *(len(prs_new) + len(nrs_new))+(1-int_flag)*nfeatures)+ self.beta * sum(~covered_new)/self.N
            T = T0**(iter/Niteration)
            alpha = np.exp(float(-obj_new +obj_curr)/T) # minimize
            if obj_new < self.maps[-1][1]:
                prs_opt,nrs_opt,obj_opt,pcovered_opt,ncovered_opt,overlap_opt,covered_opt, Yhat_opt = prs_new[:],nrs_new[:],obj_new,pcovered_new[:],ncovered_new[:],overlap_new[:],covered_new[:], Yhat_new[:]
                perror, nerror, oerror, berror = self.diagnose(pcovered_new,ncovered_new,overlap_new,covered_new,Yhat_new)
                accuracy_min = float(TP+TN)/self.N
                explainability_min = sum(covered_new)/self.N
                covered_min = covered_new
                print('\n**  max at iter = {} ** \n {}(obj) = {}(error) + {}(intepretability) + {}(exp)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(iter,round(obj_new,3),(FP+FN)/self.N, self.alpha*(len(prs_new) + len(nrs_new)), self.beta*sum(~covered_new)/self.N, (TP+TN+0.0)/self.N,sum(covered_new)/self.N,nfeatures,perror,nerror,oerror,berror ))
                self.maps.append([iter,obj_new,prs_new,nrs_new])
            
            if print_message:
                perror, nerror, oerror, berror = self.diagnose(pcovered_new,ncovered_new,overlap_new,covered_new,Yhat_new)
                print('\niter = {}, alpha = {}, {}(obj) = {}(error) + {}(intepretability) + {}(exp)\n accuracy = {}, explainability = {}, nfeatures = {}\n perror = {}, nerror = {}, oerror = {}, berror = {}\n '.format(iter,round(alpha,2),round(obj_new,3),(FP+FN)/self.N, self.alpha*(len(prs_new) + len(nrs_new)), self.beta*sum(~covered_new)/self.N, (TP+TN+0.0)/self.N,sum(covered_new)/self.N, nfeatures,perror,nerror,oerror,berror ))
                print('prs = {}, nrs = {}'.format(prs_new, nrs_new))
            if random() <= alpha:
                prs_curr,nrs_curr,obj_curr,pcovered_curr,ncovered_curr,overlap_curr,covered_curr, Yhat_curr =  prs_new[:],nrs_new[:],obj_new,pcovered_new[:],ncovered_new[:],overlap_new[:],covered_new[:], Yhat_new[:]
        self.positive_rule_set = prs_opt
        self.negative_rule_set = nrs_opt

    def diagnose(self, pcovered, ncovered, overlapped, covered, Yhat):
        perror = sum(self.Y[pcovered]!=Yhat[pcovered])
        nerror = sum(self.Y[ncovered]!=Yhat[ncovered])
        oerror = sum(self.Y[overlapped]!=Yhat[overlapped])
        berror = sum(self.Y[~covered]!=Yhat[~covered])
        return perror, nerror, oerror, berror

    def compute_obj(self,pcovered,covered):
        Yhat = np.zeros(int(self.N))
        Yhat[pcovered] = 1
        Yhat[~covered] = self.Yb[~covered] #self.Y[~covered]#
        TN,FP,FN,TP = confusion_matrix(Yhat,self.Y).ravel()
        return  Yhat,TP,FP,TN,FN

    def propose_rs(self, prs,nrs,pcovered,ncovered,overlapped, covered,Yhat, vt,print_message = False):
        incorr = np.where(Yhat[covered]!=self.Y[covered])[0]# correct interpretable models
        incorrb = np.where(Yhat[~covered]!=self.Y[~covered])[0]
        overlapped_ind = np.where(overlapped)[0]
        p = np.sum(self.pRMatrix[:,prs],axis = 1)
        n = np.sum(self.nRMatrix[:,nrs],axis = 1)
        ex = -1
        if sum(covered) ==self.N: # covering all examples.
            move = ['cut']
            self.actions.append(0)
            if len(prs)==0:
                sign = [0]
            elif len(nrs)==0:
                sign = [1]
            else:
                sign = [int(random()<0.5)]  
        elif len(incorr) ==0 and (len(incorrb)==0 or len(overlapped) ==self.N) or sum(overlapped) > sum(covered):
            self.actions.append(1)
            move = ['cut']
            sign = [int(random()<0.5)]           
        # elif (len(incorr) == 0 and (sum(covered)>0)) or len(incorr)/sum(covered) >= len(incorrb)/sum(~covered):
        #     if print_message:
        #         print(' ===== 2 ===== ')
        #     self.actions.append(2)
        #     ex = sample(list(np.where(~covered)[0]) + list(np.where(overlapped)[0]),1)[0] 
        #     if overlapped[ex] or len(prs) + len(nrs) >= (vt + self.beta)/self.alpha:
        #         # print('2')
        #         move = ['cut']
        #         sign = [int(random()<0.5)]
        #     else:
        #         # print('3')
        #         move = ['expand']
        #         sign = [int(random()<0.5)]
        else:
            # if sum(overlapped)/sum(pcovered)>.5 or sum(overlapped)/sum(ncovered)>.5:
            #     if print_message:
            #         print(' ===== 3 ===== ')
            #     # print('4')
            #     move = ['cut']
            #     sign = [int(len(prs)>len(nrs))]
            # else:  
            t = random()
            if t< 1./3: # try to decrease errors
                self.actions.append(3)
                ex = sample(list(incorr) + list(incorrb),1)[0] 
                if ex in incorr: # incorrectly classified by the interpretable model
                    rs_indicator = (pcovered[ex]).astype(int) # covered by prules
                    if random()<0.5:
                        move = ['cut']
                        sign = [rs_indicator]
                    else:
                        move = ['cut','add']
                        sign = [rs_indicator,rs_indicator]
                # elif overlapped[ex]: 
                #     if random()<0.5 :
                #         # print('5')
                #         move = ['cut']
                #         sign = [1 - self.Y[ex]]
                #     else:
                #         # print('6')
                #         move = ['cut','add']
                #         sign = [1 - self.Y[ex],1 - self.Y[ex]]
                else: # incorrectly classified by the black box model
                    move = ['add']
                    sign = [int(self.Y[ex]==1)]
            elif t<2./3: # decrease coverage
                self.actions.append(4)
                move = ['cut']
                sign = [round(random())]
            else: # increase coverage
                self.actions.append(5)
                move = ['expand']
                sign = [round(random())]
                # if random()<0.5:
                #     move.append('add')
                #     sign.append(1-rs_indicator)
                # else:
                #     move.extend(['cut','add'])
                #     sign.extend([1-rs_indicator,1-rs_indicator])
        for j in range(len(move)):
            if sign[j]==1:
                prs = self.action(move[j],sign[j],ex,prs,Yhat,pcovered)
            else:
                nrs = self.action(move[j],sign[j],ex,nrs,Yhat,ncovered)

        p = np.sum(self.pRMatrix[:,prs],axis = 1)>0
        n = np.sum(self.nRMatrix[:,nrs],axis = 1)>0
        o = np.multiply(p,n)
        return prs, nrs,p,n^o,o, np.logical_xor(p,n) + o

    def action(self,move, rs_indicator, ex, rules,Yhat,covered):
        if rs_indicator==1:
            RMatrix = self.pRMatrix
            # error = self.perror
            supp = self.psupp
        else:
            RMatrix = self.nRMatrix
            # error = self.nerror
            supp = self.nsupp
        Y = self.Y if rs_indicator else 1- self.Y
        if move=='cut' and len(rules)>0:
            """ cut """
            if random()<0.25 and ex >=0:
                candidate = list(set(np.where(RMatrix[ex,:]==1)[0]).intersection(rules))
                if len(candidate)==0:
                    candidate = rules
                cut_rule = sample(candidate,1)[0]
            else:
                p = []
                all_sum = np.sum(RMatrix[:,rules],axis = 1)
                for index,rule in enumerate(rules):
                    Yhat= ((all_sum - np.array(RMatrix[:,rule]))>0).astype(int)
                    TP,FP,TN,FN  =confusion_matrix(Yhat,Y).ravel()
                    p.append(TP.astype(float)/(TP+FP+1))
                    # p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                p = [x - min(p) for x in p]
                p = np.exp(p)
                p = np.insert(p,0,0)
                p = np.array(list(accumulate(p)))
                if p[-1]==0:
                    cut_rule = sample(rules,1)[0]
                else:
                    p = p/p[-1]
                    index = find_lt(p,random())
                    cut_rule = rules[index]
            rules.remove(cut_rule)
        elif move == 'add' and ex>=0: 
            """ add """
            score_max = -self.N *10000000
            if self.Y[ex]*rs_indicator + (1 - self.Y[ex])*(1 - rs_indicator)==1:
                # select = list(np.where(RMatrix[ex] & (error +self.alpha*self.N < self.beta * supp))[0]) # fix
                select = list(np.where(RMatrix[ex])[0])
            else:
                # select = list(np.where( ~RMatrix[ex]& (error +self.alpha*self.N < self.beta * supp))[0])
                select = list(np.where( ~RMatrix[ex])[0])
            self.select = select
            if len(select)>0:
                if random()<0.25:
                    add_rule = sample(select,1)[0]
                else: 
                    # cover = np.sum(RMatrix[(~covered)&(~covered2), select],axis = 0)
                    # =============== Use precision as a criteria ===============
                    # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                    # mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),Y[Yhat_neg_index])
                    # TP = np.sum(mat,axis = 1)
                    # FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                    # TN = np.sum(Y[Yhat_neg_index]==0)-FP
                    # FN = sum(Y[Yhat_neg_index]) - TP
                    # p = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select]
                    # add_rule = select[sample(list(np.where(p==max(p))[0]),1)[0]]
                    # =============== Use objective function as a criteria ===============
                    for ind in select:
                        z = np.logical_or(RMatrix[:,ind],Yhat)
                        TN,FP,FN,TP = confusion_matrix(z,self.Y).ravel()
                        score = FP+FN -self.beta * sum(RMatrix[~covered ,ind])
                        if score > score_max:
                            score_max = score
                            add_rule = ind
                if add_rule not in rules:
                    rules.append(add_rule)
        else: # expand
            candidates = [x for x in range(RMatrix.shape[1])]
            if rs_indicator:
                select = list(set(candidates).difference(rules))
            else:
                select = list(set(candidates).difference(rules))
            # self.error = error
            self.supp = supp
            self.select = select
            self.candidates = candidates
            self.rules = rules
            if random()<0.25:
                add_rule = sample(select, 1)[0]
            else:
                # Yhat_neg_index = np.where(np.sum(RMatrix[:,rules],axis = 1)<1)[0]
                Yhat_neg_index = np.where(~covered)[0]
                mat = np.multiply(RMatrix[Yhat_neg_index.reshape(-1,1),select].transpose(),Y[Yhat_neg_index])
                # TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                TP = np.sum(mat,axis = 1)
                FP = np.array(np.sum(RMatrix[Yhat_neg_index.reshape(-1,1),select],axis = 0) - TP)
                TN = np.sum(Y[Yhat_neg_index]==0)-FP
                FN = sum(Y[Yhat_neg_index]) - TP
                score = (FP + FN)+ self.beta * (TN + FN)
                # score = (TP.astype(float)/(TP+FP+1)) + self.alpha * supp[select] # using precision as the criteria
                add_rule = select[sample(list(np.where(score==min(score))[0]),1)[0]] 
            if add_rule not in rules:
                rules.append(add_rule)
        return rules


    def predict(self, df, Y,Yb ):
        prules = [self.prules[i] for i in self.positive_rule_set]
        nrules = [self.nrules[i] for i in self.negative_rule_set]
        # if isinstance(self.df, scipy.sparse.csc.csc_matrix)==False:
        dfn = 1-df #df has negative associations
        dfn.columns = [name.strip() + 'neg' for name in df.columns]
        df_test = pd.concat([df,dfn],axis = 1)
        if len(prules):
            p = [[] for rule in prules]
            for i,rule in enumerate(prules):
                p[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            p = (np.sum(p,axis=0)>0).astype(int)
        else:
            p = np.zeros(len(Y))
        if len(nrules):
            n = [[] for rule in nrules]
            for i,rule in enumerate(nrules):
                n[i] = (np.sum(df_test[list(rule)],axis=1)==len(rule)).astype(int)
            n = (np.sum(n,axis=0)>0).astype(int)
        else:
            n = np.zeros(len(Y))
        pind = list(np.where(p)[0])
        nind = list(np.where(n)[0])
        covered = [x for x in range(len(Y)) if x in pind or x in nind]
        Yhat = np.array([i for i in Yb])
        Yhat[nind] = 0
        Yhat[pind] = 1
        return Yhat,covered

def accumulate(iterable, func=operator.add):
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0


def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = 'neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)   
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules
