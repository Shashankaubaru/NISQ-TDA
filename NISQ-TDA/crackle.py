import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm as gaussian
from scipy.special import erfinv

import numpy as np
# import sys
# import time
import os

import gudhi as gd
import gudhi.representations as gr

# from tqdm import tqdm
# import itertools

from sklearn.metrics import auc as auc
from sklearn.metrics import roc_curve as roc_curve

from bayes_tda.intensities import RGaussianMixture
from bayes_tda.classifiers import EmpBayesFactorClassifier as EBFC
from bayes_tda.intensities import Posterior
# from bayes_tda.intensities import Posterior
# import pdb
from cmb import generate_sample_set

EPS=1e-16
pareto_alpha=3
fnl1=100
# distributions = ["gaussian", "power"]
# distributions = ["gaussian", "exponential"]
distributions = ["fnl0", "fnl1"]
mycolours = {distributions[0]: "blue", distributions[1]: "red"}


def tilt_remove_inf_PIs(single_PI_bettis):
    mySelector = gr.DiagramSelector(use=True,
                                    limit=np.inf,
                                    point_type="finite")
    transformed = []
    for PI in single_PI_bettis:
        PI = mySelector.__call__(PI)
        PI[:, 1] = PI[:, 1] - PI[:, 0]  # death - birth
        transformed.append(PI)
    return transformed


def find_bounding_box(data_array):

    if type(data_array) is list :
        if type(data_array[0]) is not np.ndarray:
            raise(ValueError("Expecting numpy arrays inside the list."))
    else:
        raise(ValueError("Expecting a list."))

    if len(data_array[0].shape) != 2:
        raise(ValueError("Expecting array to be a list of points."))

    max_range = 1 << 10
    dim = data_array[0].shape[1]
    point_from = np.ones((dim))*(max_range)
    point_to = np.ones((dim))*(-max_range)
    
    for array in data_array:
        if len(array) > 0:
            point_from = np.amin(np.vstack((array, point_from.reshape(1, -1))),
                                 axis=0)
            point_to = np.amax(np.vstack((array, point_to.reshape(1, -1))),
                               axis=0)

    if np.any(point_from > point_to):
        raise(ValueError("Error: min range greater than max."))

    return point_from.tolist(), point_to.tolist()


def bayes_factor(test_scores_array, write_folder=None,
                 title="Bayes Factor Distributions",
                 filename="bayes.png"):
    # examine score distributions

    # test_keys = list(test_scores.keys())
    # if test_keys != distributions:
    #     import pdb
    #     pdb.set_trace()
    #     print("Warning: score keys order differs from default order.")

    scores_0 = test_scores_array[0]
    scores_1 = test_scores_array[1]

    # pdb.set_trace()
    if len(scores_0.shape) < 3:
        scores_0 = scores_0.reshape(1, 2, -1)
    if len(scores_1.shape) < 3:
        scores_1 = scores_1.reshape(1, 2, -1)

    # pdb.set_trace()
    scores_0 = np.add.reduce(scores_0[:, 0, :] - scores_0[:, 1, :], axis=0)
    scores_1 = np.add.reduce(scores_1[:, 0, :] - scores_1[:, 1, :], axis=0)

    # compute aucs
    y0 = np.zeros(len(scores_0))
    y1 = np.ones(len(scores_1))

    y_true = np.concatenate([y0, y1])
    y_score = np.concatenate([scores_0, scores_1])

    tpr, fpr, _ = roc_curve(y_true, y_score)
    AUC = auc(fpr, tpr)
    
    my_bins = np.histogram_bin_edges([scores_0,
                                      scores_1], bins=50)
    plot.hist(scores_0, label=distributions[0], bins=my_bins,
              color=mycolours[distributions[0]], alpha=0.5)
    plot.hist(scores_1, label=distributions[1], bins=my_bins,
              color=mycolours[distributions[1]], alpha=0.5)
    plot.xlabel('$log p(0) / p(1) $')
    plot.ylabel('Count')
    plot.title(title + "AUC: " + str(AUC))
    plot.legend()

    if write_folder is None:
        plot.show()
    else:
        os.makedirs(write_folder, exist_ok=True)
        plot.savefig(write_folder + "/" + filename)

    plot.close()

    return AUC


def sphere(num_samples=100, dim=3, radii=None):

    x = np.random.normal(size=(num_samples, dim))
    norms = np.linalg.norm(x, axis=1)
    if radii is not None:
        x *= np.divide(radii, norms)[:, np.newaxis]
    else:
        x /= norms[:, np.newaxis]

    return x


def generate_random_points(candidate_samples=1000, num_samples=100, dim=3,
                           seed=42, min_radius=0, distribution="exponential",
                           standardise=True):
    if candidate_samples < num_samples:
        raise(ValueError("Candidate samples must be larger than num_samples."))

    if distribution == "fnl0" or distribution == "fnl1":
        sample_multiplier = candidate_samples/num_samples
        points, skipped = generate_sample_set(num_samples=num_samples,
                                              sample_multiplier=
                                              3*int(sample_multiplier),
                                              patches_to_average=1,
                                              fnl=(0
                                                   if distribution == "fnl0"
                                                   else
                                                   fnl1),
                                              nside=2048,
                                              patch_width=16,
                                              max_period_deg=2,
                                              num_periods=2,
                                              std_factor=
                                              (None if
                                               sample_multiplier == 1 else
                                               # 0.1),
                                               erfinv(1-1/sample_multiplier)/8),
                                              above_threshhold=True,
                                              norm_not_product=True,
                                              alm=False,
                                              rescale_range=False)
        print("Excluded: ", skipped)
        points = np.array(points)
    else:
        gen = np.random.default_rng(seed=seed)
        if distribution == "exponential":
            radii = gen.exponential(size=(candidate_samples))
        elif distribution == "gaussian":
            radii = gen.normal(size=(candidate_samples))
        elif distribution == "power":
            radii = gen.pareto(pareto_alpha, size=(candidate_samples))
        else:
            raise ValueError("Distribution not supported: ", distribution)

        radii = np.sort(radii)[-num_samples:]
        radii += min_radius

        points = sphere(num_samples=num_samples, dim=dim,
                        radii=radii)

    # import pdb
    # pdb.set_trace()
    if standardise:
        points_mean = points.mean(axis=0)[np.newaxis, :]
        points_std = points.std(axis=0)[np.newaxis, :]
        points = (points - points_mean) / points_std

    return points


def calc_PI(points, max_dim=1):

    rips_complex = gd.RipsComplex(points=points)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dim)
    rips_PD = simplex_tree.persistence()
    # list of list of betti number and (birth-death) tuple

    # rips_PI = [[] for _ in range(max_dim)]
    rips_PIs = []
    for dim in range(max_dim):
        betti_PI = np.array(
            simplex_tree.persistence_intervals_in_dimension(dim))
        rips_PIs.append(betti_PI)
        # rips_PIs.append(np.array([[EPS, 2*EPS]]) if len(betti_PI) == 0 else
        #                 betti_PI)

    rips_PIs = tilt_remove_inf_PIs(rips_PIs)

    return rips_PIs, rips_PD


def write_PIs_one_betti(data_PIs, data_labels,
                        filename_prefix="PI",
                        write_folder="bayesian_diagrams"):

    [xlim_from, ylim_from], [xlim_to, ylim_to] = find_bounding_box(data_PIs)

    unique_labels = set(data_labels)
    counts = {label: 0 for label in unique_labels}

    for PI, label in zip(data_PIs, data_labels):
        count_label = counts[label]
        plot.title("PI " + str(count_label) + " : " + label)
        plot.scatter(PI[:, 0], PI[:, 1], color=mycolours[label])
        if (xlim_from < xlim_to):
            plot.xlim([xlim_from, xlim_to])
        if (xlim_from < xlim_to):
            plot.ylim([ylim_from, ylim_to])
        plot.savefig(write_folder + "/" + filename_prefix + "_" + label + "_" +
                     str(count_label))
        plot.close()

        counts[label] = count_label + 1


def write_PDs_all_bettis(data_PDs, data_labels,
                         filename_prefix="PD",
                         write_folder="bayesian_diagrams"):

    unique_labels = set(data_labels)
    counts = {label: 0 for label in unique_labels}

    for PD, label in zip(data_PDs, data_labels):
        count_label = counts[label]
        gd.plot_persistence_diagram(PD)  # works on PD's only
        plot.title("PDs " + str(count_label) + " : " + label)
        # plot.scatter(PI[:, 0], PI[:, 1], color=mycolours[label])
        # if (xlim_from < xlim_to):
        #     plot.xlim([xlim_from, xlim_to])
        # if (xlim_from < xlim_to):
        #     plot.ylim([ylim_from, ylim_to])
        plot.xlim([0, 2.5])
        plot.ylim([0, 3])
        plot.savefig(write_folder + "/" + filename_prefix + "_" + label + "_" +
                     str(count_label))
        plot.close()

        counts[label] = count_label + 1


def bayesian_classification(data_PIs,
                            data_labels,
                            train_percent = 0.75,
                            filename_prefix="",
                            write_folder="bayesian_diagrams"):

    """
    Perform Bayesian Training (75% of data) and Testing (25%) on two labelled
    tilted data_PI sets (0 and 1). Returning AUC on the testing data.
    """

    # unique_labels = list(set(data_labels))
    # if unique_labels != distributions:
    #     import pdb
    #     pdb.set_trace()
    #     print("Warning: data_labels order differs from default order.")

    scores_all_betti = [[] for _ in distributions]  # outer list test_num
    for betti, PIs_betti in enumerate(data_PIs):
        [xlim_from, ylim_from], [xlim_to, ylim_to] = find_bounding_box(
            PIs_betti)

        mid_x = (xlim_to + xlim_from)/2
        mid_y = (ylim_to + ylim_from)/2
        width_x = xlim_to - xlim_from
        width_y = ylim_to - ylim_from

        mu_pri = np.array([[mid_x, mid_y]])          # prior mean
        w_pri = np.array([1])
        # prior weight, switch off with alpha=1
        sig_pri = np.array([max(width_x, width_y)])  # prior covariance mag.
        sig_data = min(width_x, width_y)/10

        # build prior and clutter
        prior = RGaussianMixture(mus=mu_pri,
                                 sigmas=sig_pri,
                                 weights=w_pri,
                                 normalize_weights=False)

        clutter = RGaussianMixture(mus=mu_pri,
                                   sigmas=sig_pri,
                                   weights=[[0]],  # switch off via weights,
                                   # better if can take None
                                   normalize_weights=False)

        classifier = EBFC(data=PIs_betti,
                          labels=data_labels,
                          data_type='diagrams')
        # if list(classifier.grouped_dgms.keys()) != distributions:
        #     import pdb
        #     pdb.set_trace()
        #     print("Warning: classifier keys order differs from default order.")

        # scores_dict, keys: labels, values: score dict (keys train) of test prob
        test_scores_one_betti = classifier.compute_scores(
            clutter,
            prior,
            prior_prop=train_percent,
            sigma_DYO=sig_data)
        # pdb.set_trace()

        # Turn dict of dict into list (test_num) of arrays (index train_num)
        scores_0, scores_1 = [test_scores_one_betti[distribution] for
                              distribution in distributions]

        scores_0 = np.array([scores_0[distribution] for distribution in
                             distributions])
        scores_1 = np.array([scores_1[distribution] for distribution in
                             distributions])

        scores_array = [scores_0, scores_1]

        if np.isinf(np.sum(scores_array)):
            import pdb
            pdb.set_trace()
        
        auc = bayes_factor(scores_array,
                           filename=filename_prefix + "_betti_" +
                           str(betti + 1),  # min_betti
                           title="Bayes Factor (Betti " + str(betti + 1)
                           + ")",
                           write_folder=write_folder)

        print('AUC for betti ' + str(betti + 1) + ' : ' + str(auc))

        for test_num in range(len(distributions)):
            scores_all_betti[test_num].append(scores_array[test_num])

    for test_num in range(len(distributions)):
        scores_all_betti[test_num] = np.array(scores_all_betti[test_num])
        # The array has shape test (2) x betti (2) x train label (2) x train (8)
    # import pdb
    # pdb.set_trace()
    auc = bayes_factor(scores_all_betti,
                       filename=filename_prefix + '_combined',
                       title="Bayes Factor (Combined)",
                       write_folder=write_folder)

    print('AUC combined: ' + str(auc))

    return auc


def bayesian_posterior(data_PIs,
                       filename_prefix="",
                       write_folder="bayesian_diagrams"):

    [xlim_from, ylim_from], [xlim_to, ylim_to] = find_bounding_box(
        data_PIs)

    mid_x = (xlim_to + xlim_from)/2
    mid_y = (ylim_to + ylim_from)/2
    width_x = xlim_to - xlim_from
    width_y = ylim_to - ylim_from

    mu_pri = np.array([[mid_x, mid_y]])          # prior mean
    w_pri = np.array([1])                        # prior weight
    sig_pri = np.array([max(width_x, width_y)])  # prior covariance magnitude

    sig_data = min(width_x, width_y)/10

    # build prior and clutter
    prior = RGaussianMixture(mus=mu_pri,
                             sigmas=sig_pri,
                             weights=w_pri,
                             normalize_weights=False)

    clutter = RGaussianMixture(mus=mu_pri,
                               sigmas=sig_pri,
                               weights=[[0]],  # switch off via weights,
                               # better if can take None
                               normalize_weights=False)

    DYO = data_PIs  # list of observed persistence diagrams
    sigma_DYO = sig_data  # 0.3 # magnitude of covariance matrix for likelihood
    posterior = Posterior(DYO, prior, clutter, sigma_DYO)

    linear_grid = np.linspace(0, 4, 28)
    # posterior.show_prior(linear_grid, cmap = 'coolwarm')
    # posterior.show_clutter(linear_grid, cmap = 'coolwarm')
    if (xlim_from < xlim_to):
        plot.xlim([xlim_from, xlim_to])
    if (xlim_from < xlim_to):
        plot.ylim([ylim_from, ylim_to*2])

    posterior.show_lambda_DYO(linear_grid, show_means=False, cmap='coolwarm')


if __name__ == "__main__":

    # top_fraction = [1]
    # top_fractions = np.arange(0.6, 0.1, -0.1)
    top_fractions = np.arange(0.6, 0.4, -0.1)
    # top_fraction = 0.5
    auc_vs_change = []
    # min_samples = 100
    num_samples = 100
    num_repeats = 20
    min_betti = 1
    max_betti = 4  # expand till, not including
    train_percent = 0.6
    dim = 3
    # dims = range(3, 20, 5)

    for top_fraction in top_fractions:
    # for dim in dims:
        candidate_samples = int(num_samples/top_fraction)

        # fraction_outside_core_exponential = np.exp(-np.log(candidate_samples))
        # fraction_outside_core_gaussian = 1 - gaussian.cdf(
        #     np.sqrt(2*np.log(candidate_samples)))
        # larger_fraction = max(fraction_outside_core_gaussian,
        #                       fraction_outside_core_exponential)
        # num_samples = max(int(larger_fraction*candidate_samples),
        #                   min_samples)

        filename_prefix = "_vs_".join(distributions) + "_num_points_" + \
            str(num_samples) + "_out_of_" + str(candidate_samples)
        write_folder = "../output/crackle_diagrams/" + filename_prefix + "_" + \
            str(dim) + "D/"
        os.makedirs(write_folder, exist_ok=True)    

        # We could try and determine outercore by significant change in betti0
        # max_betti0_diff = np.argmax(betti0[:-1] - betti0[1:])

        data_PIs_repeat = []
        data_PDs = []
        data_labels = []
        data_PIs = [[] for _ in range(max_betti)]
        collect_points = {distribution: [] for distribution in distributions}
        for distribution in distributions:
            for repeat in range(num_repeats):
                points = generate_random_points(
                    candidate_samples=candidate_samples,
                    num_samples=num_samples,
                    seed=repeat,
                    distribution=distribution,
                    dim=dim
                )
                single_PI_bettis, single_PD = calc_PI(points,
                                                      max_dim=max_betti)
                collect_points[distribution] += list(points)
                del points
                data_PIs_repeat.append(single_PI_bettis)
                data_PDs.append(single_PD)
                data_labels.append(distribution)

        lengths = {}
        for distribution in distributions:
            collect_points[distribution] = np.array(collect_points[distribution])
            lengths[distribution] = np.linalg.norm(collect_points[distribution],
                                                   axis=1).reshape(-1, 1)

        point_from = np.zeros((dim))
        point_to = np.zeros((dim))
        point_from, point_to = find_bounding_box(list(collect_points.values()))

        length_list = list(lengths.values())
        [length_from], [length_to] = find_bounding_box(length_list)
        length_bins = np.histogram_bin_edges(length_list, bins=50)

        for distribution in distributions:
            fig = plot.figure()
            ax = Axes3D(fig)
            ax.scatter(collect_points[distribution][:, 0],
                       collect_points[distribution][:, 1],
                       collect_points[distribution][:, 2],
                       s=7)
            plot.title("Point distribution (" + distribution + ")")
            plot.xlim([point_from[0], point_to[0]])
            plot.ylim([point_from[1], point_to[1]])
            ax.set_zlim(point_from[2], point_to[2])

            plot.savefig(write_folder + filename_prefix + "_sample_points_" +
                         distribution + "_" + str(dim) + "D")
            plot.close()

            plot.hist(lengths[distribution], bins=length_bins)
            plot.title("Lengths distribution (" + distribution + ")")
            plot.xlim([length_from, length_to])
            plot.ylim([0, num_samples*num_repeats//4])
            plot.savefig(write_folder + filename_prefix + "_sample_lenghts_" +
                         distribution + "_" + str(dim) + "D")
            plot.close()

        for repeat in range(num_repeats*len(distributions)):
            for betti in range(max_betti):
                data_PIs[betti].append(data_PIs_repeat[repeat][betti])

        data_PIs = data_PIs[min_betti:]  # throw away betti 0 to betti min_betti

        # write_PIs_one_betti(data_PIs[max_betti], data_labels,
        #           filename_prefix="PI_betti_" + str(max_betti),
        #           write_folder="../output/crackle_diagrams/")

        write_PDs_all_bettis(data_PDs, data_labels,
                             filename_prefix="PDs_" + filename_prefix,
                             write_folder=write_folder)

        auc_val = bayesian_classification(data_PIs,
                                          data_labels,
                                          train_percent=train_percent,
                                          filename_prefix="Bayes_Factor_" +
                                          filename_prefix,
                                          write_folder=write_folder)

        # bayesian_posterior([data_PIs[i] for i, label in enumerate(data_labels)
        #                     if label == "exponential"])

        auc_vs_change.append(auc_val)

    # Fraction
    filename_prefix = "_vs_".join(distributions) + "_num_points_" + \
        str(num_samples) + "_AUC_vs_Fraction_" + "_".join(list(map(lambda f:
                                                          "{:.2f}".format(f),
                                                          top_fractions))) + \
                                        ".png"
    write_folder = "../output/crackle_diagrams/"
    plot.plot(1-top_fractions, auc_vs_change)
    plot.title("AUC vs Fraction Core Excluded")

    # Dimension
    # filename_prefix = "_vs_".join(distributions) + "_num_points_" + \
    #     str(num_samples) + "_AUC_vs_Dims_" + "_".join(list(map(lambda f:
    #                                                      "{:d}".format(f),
    #                                                      dims))) +  ".png"
    # write_folder = "../output/crackle_diagrams/"
    # plot.plot(dims, auc_vs_change)
    # plot.title("AUC vs Dimension")

    plot.savefig(write_folder + "/" + filename_prefix)
    plot.close()
