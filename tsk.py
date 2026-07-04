import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.cluster import KMeans
from torch.nn import functional

from utils import check_tensor,reset_params
import importlib.util

def antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20):
    """

    This function run KMeans clustering to obtain the :code:`init_center` for :func:`AntecedentGMF() <AntecedentGMF>`.

    Examples
    --------
    >>> init_center = antecedent_init_center(X, n_rule=10, method="kmean", n_init=20)
    >>> antecedent = AntecedentGMF(X.shape[1], n_rule=10, init_center=init_center)


    :param numpy.array X: Feature matrix with the size of :math:`[N,D]`, where :math:`N` is the
        number of samples, :math:`D` is the number of features.
    :param numpy.array y: None, not used.
    :param int n_rule: Number of rules :math:`R`. This function will run a KMeans clustering to
        obtain :math:`R` cluster centers as the initial antecedent center for TSK modeling.
    :param str method: Current version only support "kmean".
    :param str engine: "sklearn" or "faiss". If "sklearn", then the :code:`sklearn.cluster.KMeans()`
        function will be used, otherwise the :code:`faiss.Kmeans()` will be used. Faiss provide a
        faster KMeans clustering algorithm, "faiss" is recommended for large datasets.
    :param int n_init: Number of initialization of the KMeans algorithm. Same as the parameter
        :code:`n_init` in :code:`sklearn.cluster.KMeans()` and the parameter :code:`nredo` in
        :code:`faiss.Kmeans()`.
    """
    def faiss_cluster_center(X, y=None, n_rule=2, n_init=20):
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=n_rule, nredo=n_init)
        km.train(np.ascontiguousarray(X.astype("float32")))
        centers = km.centroids.T
        return centers

    if method == "kmean":
        if engine == "faiss":
            package_name = "faiss"
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                center = faiss_cluster_center(X=X, y=y, n_rule=n_rule)
                return center
            else:
                print("Package " + package_name + " is not installed, running scikit-learn KMeans...")
        km = KMeans(n_clusters=n_rule, n_init=n_init)
        km.fit(X)
        return km.cluster_centers_.T


class Antecedent(nn.Module):
    def forward(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError


class AntecedentGMF(Antecedent):
    """

    Parent: :code:`torch.nn.Module`

    The antecedent part with Gaussian membership function. Input: data, output the corresponding
    firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the
    :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = \exp(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}),\\
        &f_{r}(\mathbf{x})=\prod_{d=1}^{D}\mu_{r,d}(x_d),\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.


    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`,
        HTSK is used. Otherwise the original defuzzification is used. More details can be found at [1].
        TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True` is highly
         recommended for any-dimensional problems.
    :param numpy.array init_center: Initial center of the Gaussian membership function with
        the size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
        :code:`init_center` as the obtained centers. You can simply run
        :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
        to obtain the center.
    :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
    :param float eps: A constant to avoid the division zero error.
    """
    def __init__(self, in_dim, n_rule, high_dim=False, init_center=None, init_sigma=1., eps=1e-8):
        super(AntecedentGMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_sigma = init_sigma
        self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))
        self.sigma = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))

        self.reset_parameters()

    def init(self, center, sigma):
        """

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the
            size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
            :code:`init_center` as the obtained centers. You can simply run
            :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
            to obtain the center.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.
        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.

        :return:
        """
        init.constant_(self.sigma, self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        """

        Forward method of Pytorch Module.

        :param torch.tensor X: Pytorch tensor with the size of :math:`[N, D]`, where :math:`N` is the number of samples, :math:`D` is the input dimension.
        :return: Firing level matrix :math:`U` with the size of :math:`[N, R]`.
        """
        frs = self.zr_op(
            -(X.unsqueeze(dim=2) - self.center) ** 2 * (0.5 / (self.sigma ** 2 + self.eps)), dim=1
        )
        frs = functional.softmax(frs, dim=1)
        return frs

class TSK(nn.Module):
    """

    Parent: :code:`torch.nn.Module`

    This module define the consequent part of the TSK model and combines it with a pre-defined
     antecedent module. The input of this module is the raw feature matrix, and output
     the final prediction of a TSK model.

    :param int in_dim: Number of features :math:`D`.
    :param int out_dim: Number of output dimension :math:`C`.
    :param int n_rule: Number of rules :math:`R`, must equal to the :code:`n_rule` of
        the :code:`Antecedent()`.
    :param torch.Module antecedent: An antecedent module, whose output dimension should be
        equal to the number of rules :math:`R`.
    :param int order: 0 or 1. The order of TSK. If 0, zero-oder TSK, else, first-order TSK.
    :param float eps: A constant to avoid the division zero error.
    :param torch.nn.Module consbn: If none, the raw feature will be used as the consequent input;
        If a pytorch module, then the consequent input will be the output of the given module.
        If you wish to use the BN technique we mentioned in
        `Models & Technique <../models.html#batch-normalization>`_,  you can set
        :code:`precons=nn.BatchNorm1d(in_dim)`.
    """
    def __init__(self, in_dim, out_dim, n_rule, antecedent, order=1, eps=1e-8, precons=None):
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rule = n_rule
        self.antecedent = antecedent
        self.precons = precons

        self.order = order
        assert self.order == 0 or self.order == 1, "Order can only be 0 or 1."
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        if self.order == 0:
            self.cons = nn.Linear(self.n_rule, self.out_dim, bias=True)
        else:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rule, self.out_dim)

    def reset_parameters(self):
        """
        Re-initialize all parameters, including both consequent and antecedent parts.

        :return:
        """
        reset_params(self.antecedent)
        self.cons.reset_parameters()

        if self.precons is not None:
            self.precons.reset_parameters()

    def forward(self, X, get_frs=False):
        """

        :param torch.tensor X: Input matrix with the size of :math:`[N, D]`,
            where :math:`N` is the number of samples.
        :param bool get_frs: If true, the firing levels (the output of the antecedent)
            will also be returned.

        :return: If :code:`get_frs=True`, return the TSK output :math:`Y\in \mathbb{R}^{N,C}`
            and the antecedent output :math:`U\in \mathbb{R}^{N,R}`. If :code:`get_frs=False`,
            only return the TSK output :math:`Y`.
        """
        frs = self.antecedent(X)

        if self.precons is not None:
            X = self.precons(X)

        if self.order == 0:
            cons_input = frs
        else:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rule, X.size(1)])  # [n_batch, n_rule, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            cons_input = torch.cat([X, frs], dim=1)

        output = self.cons(cons_input)
        if get_frs:
            return output, frs
        return output
