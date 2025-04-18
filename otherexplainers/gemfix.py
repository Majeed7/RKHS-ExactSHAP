import shap
from shap.utils._legacy import IdentityLink
from math import comb , factorial
from scipy.linalg import cholesky, cho_solve
import numpy as np 

from shap.explainers._kernel import * #Kernel

from shap.utils._legacy import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data
from shap.utils._legacy import convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData, SparseData
from shap.utils import safe_isinstance
from scipy.special import binom
from scipy.sparse import issparse
import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
import warnings
import gc
from sklearn.linear_model import LassoLarsIC, Lasso, lars_path, lasso_path, LarsCV, LassoCV
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from tqdm.auto import tqdm

log = logging.getLogger('GEMFIX')

import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import NuSVR
from itertools import combinations
import heapq


class GEMFIX(KernelExplainer):

    def __init__(self, model, data, link=IdentityLink(), **kwargs):

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        self.data = convert_to_data(data, keep_index=self.keep_index)
        self.lam = kwargs.get("lam", 0.001)
        model_null = match_model_to_data(self.model, self.data)

        # enforce our current input type limitations
        assert isinstance(self.data, DenseData) or isinstance(self.data, SparseData), \
               "Shap explainer only supports the DenseData and SparseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData or SparseData currently."

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning("Using " + str(len(self.data.weights)) + " background data samples could cause " +
                        "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to " +
                        "summarize the background as K samples.")

        # init our parameters
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        if safe_isinstance(model_null, "tensorflow.python.framework.ops.EagerTensor"):
            model_null = model_null.numpy()
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    def shap_values(self, X, **kwargs):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "auto" (default for now, but deprecated), "aic", "bic", or float
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
            space is enumerated, otherwise it uses no regularization. THE BEHAVIOR OF "auto" WILL CHANGE
            in a future version to be based on num_features instead of AIC.
            The "aic" and "bic" options use the AIC and BIC rules for regularization.
            Using "num_features(int)" selects a fix number of top features. Passing a float directly sets the
            "alpha" parameter of the sklearn.linear_model.Lasso model used for feature selection.
            
        gc_collect : bool
           Run garbage collection after each explanation round. Sometime needed for memory intensive explanations (default False).

        Returns
        -------
        array or list
            For models with a single output this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored as expected_value
            attribute of the explainer). For models with vector outputs this returns a list
            of such matrices, one for each output.
        """
        self.lam = kwargs.get("lam", 0.001)

        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            if self.keep_index:
                index_value = X.index.values
                index_name = X.index.name
                column_name = list(X.columns)
            X = X.values

        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tolil()
        assert x_type.endswith(arr_type) or sp.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        # single instance
        if len(X.shape) == 1:
            data = X.reshape((1, X.shape[0]))
            if self.keep_index:
                data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.shape
            if len(s) == 2:
                outs = [np.zeros(s[0]) for j in range(s[1])]
                for j in range(s[1]):
                    outs[j] = explanation[:, j]
                return outs

            # single-output
            else:
                out = np.zeros(s[0])
                out[:] = explanation
                return out

        # explain the whole dataset
        elif len(X.shape) == 2:
            explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
                data = X[i:i + 1, :]
                if self.keep_index:
                    data = convert_to_instance_with_index(data, column_name, index_value[i:i + 1], index_name)
                explanations.append(self.explain(data, **kwargs))
                if kwargs.get("gc_collect", False):
                    gc.collect()

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0])) for j in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i] = explanations[i][:, j]
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i]
                return out

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        self.lam = kwargs.get("lam", 0.001)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            # weight_vector = np.ones(len(weight_vector))
            # total_counts = np.array([binom(self.P,i).astype(int) for i in range(1,self.P)])
            # weight_vector = np.arange(1, self.P) #total_counts / total_counts.sum()
            # weight_vector = weight_vector / np.sum(weight_vector)

            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))

            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()





            #########################################################################
            ### GEMFIX sampling strategy
            #########################################################################
            # #weights = np.arange(1, self.P)  # Weights from 1 to n-1
            
            # # possibel counts of subsets for each size
            # total_counts = np.array([binom(self.P,i).astype(int) for i in range(1,self.P)])
            
            # # the weight vector is proportionate to the total counts of each subset size
            # normalized_weights = total_counts / total_counts.sum()
            
            # # Calculate the number of samples for each subset size
            # sample_counts = np.floor(normalized_weights * self.nsamples).astype(int)
            
            # # Initialize the matrix
            # rows_accumulated = 0
            # matrix_list = []
            
            # for subset_size, count in enumerate(sample_counts, 1):  # start with subset_size 1 to n-1
            #     for _ in range(count):
            #         # Generate a binary vector with 'subset_size' number of 1's
            #         vector = np.zeros(self.P, dtype=int)
            #         ones_indices = np.random.choice(self.P, subset_size, replace=False)
            #         vector[ones_indices] = 1
            #         matrix_list.append(vector)

            #         self.addsample(instance.x, vector, 0)
                
            #     rows_accumulated += count
            
            # # If rounding caused us to have too few or too many samples, adjust accordingly
            # if rows_accumulated < self.nsamples:
            #     # Add additional samples randomly
            #     additional_rows = self.nsamples - rows_accumulated
            #     for _ in range(additional_rows):
            #         subset_size = np.random.choice(np.arange(1, self.P), p=normalized_weights)
            #         vector = np.zeros(self.P, dtype=int)
            #         ones_indices = np.random.choice(self.P, subset_size, replace=False)
            #         vector[ones_indices] = 1
            #         matrix_list.append(vector)
            #         self.addsample(instance.x, vector, 0)

            # elif rows_accumulated > self.nsamples:
            #     # Remove extra samples randomly
            #     matrix_list = np.random.permutation(matrix_list)[:self.nsamples]
            
            # # Convert list of arrays into a single numpy array
            # #matrix = np.array(matrix_list)

            
            # m = self.maskMatrix
            # mmT = m @ m.T
            # nonzeros = m @ np.ones((m.shape[1], ))
            # find_index = mmT == nonzeros  
            # np.fill_diagonal(find_index, False)
            # print(find_index == True)



            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        return phi

    @staticmethod
    def not_equal(i, j):
        number_types = (int, float, np.number)
        if isinstance(i, number_types) and isinstance(j, number_types):
            return 0 if np.isclose(i, j, equal_nan=True) else 1
        else:
            return 0 if i == j else 1

    def varying_groups(self, x):
        if not sp.sparse.issparse(x):
            varying = np.zeros(self.data.groups_size)
            for i in range(0, self.data.groups_size):
                inds = self.data.groups[i]
                x_group = x[0, inds]
                if sp.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.data.data[:, inds]))
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            varying_indices = []
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
            remove_unvarying_indices = []
            for i in range(0, len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = self.data.data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if sp.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not \
                        (np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):
        if sp.sparse.issparse(self.data.data):
            # We tile the sparse matrix in csr format but convert it to lil
            # for performance when adding samples
            shape = self.data.data.shape
            nnz = self.data.data.nnz
            data_rows, data_cols = shape
            rows = data_rows * self.nsamples
            shape = rows, data_cols
            if nnz == 0:
                self.synth_data = sp.sparse.csr_matrix(shape, dtype=self.data.data.dtype).tolil()
            else:
                data = self.data.data.data
                indices = self.data.data.indices
                indptr = self.data.data.indptr
                last_indptr_idx = indptr[len(indptr) - 1]
                indptr_wo_last = indptr[:-1]
                new_indptrs = []
                for i in range(0, self.nsamples - 1):
                    new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
                new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
                new_indptr = np.concatenate(new_indptrs)
                new_data = np.tile(data, self.nsamples)
                new_indices = np.tile(indices, self.nsamples)
                self.synth_data = sp.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=shape).tolil()
        else:
            self.synth_data = np.tile(self.data.data, (self.nsamples, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded * self.N  # N: number of samples in the training set
        if isinstance(self.varyingFeatureGroups, (list,)):
            for j in range(self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.synth_data[offset:offset+self.N, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    self.synth_data[offset:offset+self.N, group] = x[0, group]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if sp.sparse.issparse(x) and not sp.sparse.issparse(self.synth_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_data[offset:offset+self.N, groups] = evaluation_data
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun*self.N:self.nsamplesAdded*self.N,:]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun*self.N:self.nsamplesAdded*self.N]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1

    def solve(self, fraction_evaluated, dim):
        lam = self.lam 

        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        mat = self.maskMatrix.copy()
        mat = np.append(mat, np.array([np.ones((self.P)), np.zeros((self.P,))]), axis=0)
        inner_prod = mat @ mat.T
        sample_weights = np.append(np.ones((self.nsamples,)), [2000000000]*2, axis=0)
        Omega = (2**inner_prod - 1) + lam * np.diag(sample_weights) 
        
        y_hat = np.append(eyAdj, (self.fx[dim] - self.link.f(self.fnull[dim]), self.link.f(self.fnull[dim])))
        self.Omega = Omega 
        self.y_hat = y_hat
        self.mat = mat
        self.sample_weights = sample_weights
        self.inner_prod = inner_prod
        
        sample_set_size = np.array(np.sum(mat, 1), dtype=int) 
        size_weight = np.zeros((self.P,))
        for i in range(1,self.P+1):
            for j in range(1,i+1):
                size_weight[i-1] += (1/j) * comb(i-1,j-1)
        
        alpha_weight = np.array([size_weight[t-1] if t != 0 else 0 for t in sample_set_size])
        
        self.size_weight = size_weight
        self.sample_set_size = sample_set_size
        self.alpha_weight = alpha_weight
        
        ## closed form solution without regularizing the dual problem  
        L = cholesky(Omega, lower=True)
        alpha = cho_solve((L, True), y_hat)

        shapley_val = np.zeros((self.P,))
        for i in range(self.P):
            shapley_val[i] = (alpha_weight * mat[:,i]) @ alpha

        #approx_interactions = self.approximate_interaction_detection(self.maskMatrix, alpha[:-2])
        #interactions_mobius = self.interaction_detection()
        #shapley_val = self.sparse_shapley_value()
        for i in range(self.M):
            if np.abs(shapley_val[i]) < 1e-5:
                shapley_val[i] = 0
        
        return shapley_val, np.ones(len(shapley_val))


    ## This is a function for detecting interactions with solving L1 regularization problem
    def interaction_detection(self):
        ### First, identifying the most important interacting terms 
        model = lars_path(self.Omega, self.y_hat, method='lasso')
        coefs = model[2] ## coefficient of the models for different lambda' values

        # select 6 alpha and add the interacting terms as potential interations of the game
        solution_index = (coefs.shape[1] * np.array([0.05, .1, .3])).astype(int) #np.array([0.05, .1, .2, .3, .4, .5])).astype(int) 
        unique_interactions = set()
        adj_mat = np.zeros((self.P, self.P))
        for ind in solution_index:
            ## add all the possible interactions given the feature size
            interactions, approx_int, max_order = self.interactions_from_alpha(self.mat[:-2,:], coefs[:-2,ind])
            intc_topitems = heapq.nlargest(100, interactions, key=lambda x: abs(x[1])) #interactions.sort(key=lambda x: np.abs(x[1]), reverse=True)
            
            coexist_mat_adjusted = self.mat[:-2,:] * (coefs[:-2,ind][:,np.newaxis])
            adj_mat += (coexist_mat_adjusted.T @ coexist_mat_adjusted)

            for elem in intc_topitems: ## only the first 100 interactions to be added
                unique_interactions.add(elem[0])

            ## if approx_int is True, it means that we need to add approximating interacting terms
            if False:
                approx_terms = self.approximate_interaction_from_alpha(self.mat[:-2,:], coefs[:-2,ind])

                for int_term in approx_terms:
                    if len(int_term) > max_order: 
                        unique_interactions.add(frozenset(int_term))
                        continue
                    
                    for size in range(len(int_term)):
                        for combo in combinations(int_term, size):
                            # Create a set representing the subset
                            subset_set = frozenset(combo)
                            
                            # Add the value to this subset key in the dictionary
                            unique_interactions.add(subset_set)

            

        for i in range(self.P):
            unique_interactions.add(frozenset({i}))
        unique_interactions = list(unique_interactions)
        unique_interactions.sort(key=lambda x: (len(x), (list(x)[0])))

        ## Second, assigning a value for interacting terms: this is done by...
        ## add the interacting terms to the self.mat matrix and run a lasso regression to find their contribution 
        extended_mat = np.zeros( (self.mat.shape[0], len(unique_interactions)) )
        for i, feature_set in enumerate(unique_interactions):
            extended_mat[:,i] = np.prod(self.mat[:,list(feature_set)], axis=1)
        
        model_extended = LassoLarsIC(criterion='bic').fit(extended_mat, self.y_hat)
        nonzero_index = np.nonzero(model_extended.coef_)[0]
        nonzero_coef = model_extended.coef_[nonzero_index]
        nonzero_interact  = [unique_interactions[i] for i in nonzero_index]

        self.sparse_coef = list(zip(nonzero_interact, nonzero_coef))

        selected_item = [item for item in nonzero_interact if len(item) > 1] ## only get the interaction effects, not the main ones

        selected_interactions = list(zip(selected_item, model_extended.coef_[nonzero_index]))
        selected_interactions.sort(key=lambda x: abs(x[1]), reverse=True)

        return selected_interactions
    
    def approximate_interaction_from_alpha(self, matrix, alpha):
        interactions = []
        import networkx as nx
        from cdlib import algorithms
        #alpha_ind = np.where(abs(alpha) > np.mean(abs(alpha)))
        alpha_ind = np.where(abs(alpha) > np.mean(abs(alpha)))[0]
        mat = matrix[alpha_ind,:]
        adj_mat = mat.T @ mat
        G = nx.from_numpy_array(adj_mat)
        communities = algorithms.leiden(G) 
        interactions += list(frozenset(c) for c in nx.find_cliques(G) if len(c) > 1)
        [interactions.append(frozenset(cm)) for cm in communities.communities if len(cm) > 1]

        return interactions
    
    def interactions_from_alpha(self, matrix, values):
        interactions_dict = {}
        
        # Process each row in the matrix along with its corresponding value
        for row, value in zip(matrix, values):
            if abs(value) < np.mean(np.abs(values)) and abs(value) < 1e-5: continue 

            n = len(row)
            indices = [i for i in range(n) if row[i] == 1]

            max_order = len(indices)
            approx_int = False
            if len(indices) > 15 and len(indices) <= 40:
                max_order = 10
                approx_int = True
            elif len(indices) > 40:
                max_order = 3
                approx_int = True
                
            
            
            # Generate all subsets for the indices where the row has 1s
            #for size in range(2, len(indices) + 1):
            for size in range(2, max_order + 1):
                for combo in combinations(indices, size):
                    # Create a set representing the subset
                    subset_set = frozenset(combo)
                    
                    # Add the value to this subset key in the dictionary
                    if subset_set in interactions_dict:
                        interactions_dict[subset_set] += value
                    else:
                        interactions_dict[subset_set] = value

        return list(interactions_dict.items()), approx_int, max_order
    

    def sparse_shapley_value(self):
        if self.sparse_coef == None:
            interactions = self.interaction_detection


        shapley_val = np.zeros((self.P,))
        for i in range(self.P):
            shapley_val[i] = np.sum([t[1] for t in self.sparse_coef if i in t[0]])

        return shapley_val
