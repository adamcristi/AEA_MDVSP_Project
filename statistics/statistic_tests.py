import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import pingouin as pg
import copy
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
np.seterr(divide='raise')


def read_data(filename):
    with open(filename) as fd:
        first_line = fd.readline()
        used_methods = [method.strip() for method in first_line.replace("\n", "").split(";")]
        used_instances = []
        means_instances = []

        line = fd.readline()
        while line:
            used_instance, means_instance = line.replace("\n", "").split(";", 1)
            used_instances.append(used_instance)

            values_means = [float(value.replace(" ", "").replace(",", ".")) for value in means_instance.split(";") if value != '']
            means_instances.append(values_means)

            line = fd.readline()

    return used_methods, used_instances, np.array(means_instances)


def process_data(used_methods, used_instances, means_instances):
    data_df = np.concatenate((np.array([used_instances]).T, means_instances), axis=1)
    columns_df = ["data_instance"] + used_methods

    df = pd.DataFrame(data=data_df, columns=columns_df)
    return df


# One-Way Statistics

def kruskal_wallis_h_test(df):
    null_hypothesis_H0 = "the median is equal across all methods"
    alternative_hypothesis_Ha = "the median is not equal across all methods"
    threshold_kruskal_wallis_test = 0.05

    statistic_kruskal, pvalue_kruskal = stats.kruskal(df['acs_partitioned_graph'], df['acs_whole_graph_sh'], df['acs_whole_graph_gh'], df['pso'])
    print("Kruskal-Wallis H Test ->", end=" ")
    print("statistic = {:.5f}, pvalue = {:.9f}".format(statistic_kruskal, pvalue_kruskal))
    if pvalue_kruskal < threshold_kruskal_wallis_test:
        print("The null hypothesis ({}) is rejected".format(null_hypothesis_H0), end="\n\n")
        return "rejected"
    else:
        print("The null hypothesis ({}) cannot be rejected".format(null_hypothesis_H0), end="\n\n")
        return "unrejected"


def posthoc_dunn_test(df):
    threshold_dunn_test = 0.05

    data_dunn_test = []
    for column in df.columns:
        if column != 'data_instance':
            data_dunn_test.append(df[column].to_numpy())
    data_dunn_test = np.array(data_dunn_test)

    pvalues_dunn = sp.posthoc_dunn(data_dunn_test, p_adjust='bonferroni')
    pvalues_dunn.columns = df.columns[1:]
    pvalues_dunn.index = df.columns[1:]
    print("Dunn Test")
    print(pvalues_dunn, end="\n\n")

    pos_significant_methods = np.where(pvalues_dunn.to_numpy() < threshold_dunn_test)
    rows_signigicant_methods = pos_significant_methods[0]
    cols_signigicant_methods = pos_significant_methods[1]
    for index, row in enumerate(rows_signigicant_methods):
        col = cols_signigicant_methods[index]
        if row < col:
            print("Between the methods {} and {} is a statistically significant difference with pvalue = {:.6f}".format(df.columns[row+1],
                                                                                                                        df.columns[col+1],
                                                                                                                        pvalues_dunn.iloc[row][col]))
    print()


# Two-Way Statistics

def friedman_test_chi_squared_distribution(df):  # ? p-value valid numai daca numarul de metode > 10 ?
    null_hypothesis_H0 = "all the methods have the same average ranking / probability distribution"
    alternative_hypothesis_Ha = "at least two methods doesn't have the same average ranking / probability distribution"
    threshold_friedman_test_chisquare = 0.05

    statistic_friedman_chisquare, pvalue_friedman_chisquare = stats.friedmanchisquare(df['acs_partitioned_graph'], df['acs_whole_graph_sh'], df['acs_whole_graph_gh'], df['pso'])
    print("Friedman Test based on Chi Square Distribution ->", end=" ")
    print("statistic = {:.5f}, pvalue = {:.9f}".format(statistic_friedman_chisquare, pvalue_friedman_chisquare))
    if pvalue_friedman_chisquare < threshold_friedman_test_chisquare:
        print("The null hypothesis ({}) is rejected".format(null_hypothesis_H0), end="\n\n")
        return "rejected"
    else:
        print("The null hypothesis ({}) cannot be rejected".format(null_hypothesis_H0), end="\n\n")
        return "unrejected"


def friedman_test_f_distribution(df):
    null_hypothesis_H0 = "all the methods have the same probability distribution / average ranking"
    alternative_hypothesis_Ha = "at least two methods doesn't have the same probability distribution / average ranking"
    threshold_friedman_test_f = 0.05

    df_friedman_test_f = copy.deepcopy(df)
    df_friedman_test_f.drop('data_instance', axis=1, inplace=True)
    df_friedman_test_f = df_friedman_test_f.astype(float)

    try:
        statistic_result = pg.friedman(df_friedman_test_f, method="f")
    except ZeroDivisionError:
        return "not computed"
    except FloatingPointError:
        return "not computed"

    statistic_friedman_f = statistic_result['F'][0]
    pvalue_friedman_f = statistic_result['p-unc'][0]

    print("Friedman Test based on F Distribution ->", end=" ")
    print("statistic = {:.5f}, pvalue = {:.6f}".format(statistic_friedman_f, pvalue_friedman_f))
    if pvalue_friedman_f < threshold_friedman_test_f:
       print("The null hypothesis ({}) is rejected".format(null_hypothesis_H0), end="\n\n")
       return "rejected"
    else:
       print("The null hypothesis ({}) cannot be rejected".format(null_hypothesis_H0), end="\n\n")
       return "unrejected"


def friedman_test_f_distribution_paper(df):
    null_hypothesis_H0 = "all the methods have the same probability distribution / average ranking"
    alternative_hypothesis_Ha = "at least two methods doesn't have the same probability distribution / average ranking"
    threshold_friedman_test_f = 0.05

    df_friedman_test_f = copy.deepcopy(df)
    df_friedman_test_f.drop('data_instance', axis=1, inplace=True)
    df_friedman_test_f = df_friedman_test_f.astype(float)

    num_instances = df_friedman_test_f.shape[0]
    num_methods = df_friedman_test_f.shape[1]

    ranks = np.zeros((num_instances, num_methods))

    for index_instance in range(num_instances):
        means_all_methods_current_instance = df_friedman_test_f.iloc[[index_instance]].to_numpy()[0]

        pos_sorted_means = np.argsort(means_all_methods_current_instance)

        ties = 0
        current_rank = 1
        for index_pos, pos in enumerate(pos_sorted_means):
            if index_pos < len(pos_sorted_means) - 1 and \
                    means_all_methods_current_instance[pos] == means_all_methods_current_instance[pos_sorted_means[index_pos + 1]]:
                ties += 1
            else:
                if ties != 0:
                    current_rank += np.mean(np.arange(current_rank + 1, current_rank + ties + 1))
                    while ties >= 0:
                        ranks[index_instance][pos_sorted_means[index_pos - ties]] = current_rank
                        ties -= 1

                    current_rank = np.ceil(current_rank)
                    ties = 0
                else:
                    ranks[index_instance][pos] = current_rank
                    current_rank += 1

        if ties != 0:
            current_rank += np.mean(np.arange(current_rank + 1, current_rank + ties + 1))
            while ties >= 0:
                ranks[index_instance][pos_sorted_means[len(pos_sorted_means) - 1 - ties]] = current_rank
                ties -= 1

    ranks = ranks.T

    a2 = np.sum(np.square(ranks))
    b2 = 1 / num_instances * np.sum(np.square(np.sum(ranks, axis=1)))

    try:
        statistic_friedman_f = (num_instances - 1) * (b2 - num_instances * num_methods * (num_methods + 1) ** 2 / 4) / (a2 - b2)
    except FloatingPointError:
        return "not computed"

    value_distribution_f = stats.f.ppf(1 - threshold_friedman_test_f, num_methods - 1, (num_instances - 1) * (num_instances - 1))

    print(statistic_friedman_f)
    print(value_distribution_f)

    print("Friedman Test based on F Distribution from Paper ->", end=" ")
    print("statistic = {:.5f}, value f distribution = {:.6f}".format(statistic_friedman_f, value_distribution_f))
    if value_distribution_f < statistic_friedman_f:
        print("The null hypothesis ({}) is rejected".format(null_hypothesis_H0), end="\n\n")
        return "rejected"
    else:
        print("The null hypothesis ({}) cannot be rejected".format(null_hypothesis_H0), end="\n\n")
        return "unrejected"


def posthoc_nemenyi_friedman_test(df):
    threshold_nemenyi_friedman_test = 0.05

    data_nemenyi_friedman_test = []
    for column in df.columns:
        if column != 'data_instance':
            data_nemenyi_friedman_test.append(df[column].to_numpy())
    data_nemenyi_friedman_test = np.array(data_nemenyi_friedman_test)

    pvalues_nemenyi_friedman = sp.posthoc_nemenyi_friedman(data_nemenyi_friedman_test.T)
    pvalues_nemenyi_friedman.columns = df.columns[1:]
    pvalues_nemenyi_friedman.index = df.columns[1:]
    print("Nemenyi Friedman Test")
    print(pvalues_nemenyi_friedman, end="\n\n")

    pos_significant_methods = np.where(pvalues_nemenyi_friedman.to_numpy() < threshold_nemenyi_friedman_test)
    rows_signigicant_methods = pos_significant_methods[0]
    cols_signigicant_methods = pos_significant_methods[1]
    for index, row in enumerate(rows_signigicant_methods):
        col = cols_signigicant_methods[index]
        if row < col:
            print("Between the methods {} and {} is a statistically significant difference with pvalue = {:.6f}".format(df.columns[row+1],
                                                                                                                        df.columns[col+1],
                                                                                                                        pvalues_nemenyi_friedman.iloc[row][col]))
    print()


def posthoc_paired_comparisons_paper(df):
    print("Paired Comparison from Paper")
    significance_level = 0.05

    df_paired_comparisons = copy.deepcopy(df)
    df_paired_comparisons.drop('data_instance', axis=1, inplace=True)
    df_paired_comparisons = df_paired_comparisons.astype(float)

    num_instances = df_paired_comparisons.shape[0]
    num_methods = df_paired_comparisons.shape[1]

    ranks = np.zeros((num_instances, num_methods))

    for index_instance in range(num_instances):
        means_all_methods_current_instance = df_paired_comparisons.iloc[[index_instance]].to_numpy()[0]

        pos_sorted_means = np.argsort(means_all_methods_current_instance)

        ties = 0
        current_rank = 1
        for index_pos, pos in enumerate(pos_sorted_means):
            if index_pos < len(pos_sorted_means) - 1 and \
                    means_all_methods_current_instance[pos] == means_all_methods_current_instance[pos_sorted_means[index_pos + 1]]:
                ties += 1
            else:
                if ties != 0:
                    current_rank += np.mean(np.arange(current_rank + 1, current_rank + ties + 1))
                    while ties >= 0:
                        ranks[index_instance][pos_sorted_means[index_pos - ties]] = current_rank
                        ties -= 1

                    current_rank = np.ceil(current_rank)
                    ties = 0
                else:
                    ranks[index_instance][pos] = current_rank
                    current_rank += 1

        if ties != 0:
            current_rank += np.mean(np.arange(current_rank + 1, current_rank + ties + 1))
            while ties >= 0:
                ranks[index_instance][pos_sorted_means[len(pos_sorted_means) - 1 - ties]] = current_rank
                ties -= 1

    ranks = ranks.T

    a2 = np.sum(np.square(ranks))
    b2 = 1 / num_instances * np.sum(np.square(np.sum(ranks, axis=1)))

    threshold_comparison = np.sqrt(2 * num_instances * (a2 - b2) / ((num_instances - 1) * (num_methods - 1)))
    value_distribution_t = stats.t.ppf(1 - significance_level / 2, (num_instances - 1) * (num_methods - 1))
    threshold_comparison *= value_distribution_t

    summed_ranks = np.sum(ranks, axis=1).T

    values_comparisons = np.zeros((num_methods, num_methods))
    for index1 in range(summed_ranks.shape[0]-1):
        for index2 in range(index1 + 1, summed_ranks.shape[0]):
            values_comparisons[index1][index2] = values_comparisons[index2][index1] = np.abs(summed_ranks[index1] - summed_ranks[index2])
    df_comparisons = pd.DataFrame(values_comparisons, index=df.columns[1:], columns=df.columns[1:])
    print(df_comparisons, end="\n\n")

    for index1 in range(summed_ranks.shape[0]-1):
        for index2 in range(index1 + 1, summed_ranks.shape[0]):
            if np.abs(summed_ranks[index1] - summed_ranks[index2]) > threshold_comparison:
                print("Between the methods {} and {} is a statistically significant difference with value = {:}".format(df.columns[index1 + 1],
                                                                                                                        df.columns[index2 + 1],
                                                                                                                        np.abs(summed_ranks[index1] - summed_ranks[index2])))
    print()


def plot_visualization(df):
    df_melt = copy.deepcopy(df)
    df_melt = pd.melt(df_melt, id_vars=['data_instance'], value_vars=df.columns[1:])
    df_melt.columns = ['data_instance', 'methods', 'value']
    df_melt["value"] = df_melt["value"].astype(float)

    fig = plt.figure(figsize=(10,11))
    ax = sns.boxplot(x="data_instance", y='value', data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="data_instance", y="value", hue="methods", data=df_melt, palette=['yellow', 'red', 'green', 'blue'])

    ax.set_ylabel("Medie 30 Rulări", fontsize=13)
    ax.set_xlabel("Instanțe", fontsize=13)
    ax.tick_params(axis='x', rotation=30)

    new_labels=['ACS Graf Partiționat', 'ACS Graf Întreg ES', 'ACS Graf Întreg EG', 'PSO']
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=new_labels, title="Metode Utilizate")

    plt.title("Comparație Rezultate Obținute", fontsize=16, y=1.05)
    plt.savefig("statistic_plot.png")


if __name__ == "__main__":
    methods, instances, means = read_data(filename="results.txt")
    df_means = process_data(methods, instances, means)
    print(df_means, end="\n\n")

    if kruskal_wallis_h_test(df_means) == "rejected":
        print("Significant Kruskal-Wallis H Test => Post Hoc Testing")
        posthoc_dunn_test(df_means)

    # chi square distribution; from scipy library
    if friedman_test_chi_squared_distribution(df_means) == "rejected":
        print("Significant Friedman Test based on Chi Square Distribution => Post Hoc Testing")
        posthoc_nemenyi_friedman_test(df_means)
        print("\nSignificant Friedman Test based on Chi Square Distribution => Post Hoc Testing")
        posthoc_paired_comparisons_paper(df_means)

    # f distribution; from pingouin library
    if friedman_test_f_distribution(df_means) == "rejected":
        print("Significant Friedman Test based on F Distribution => Post Hoc Testing")
        posthoc_nemenyi_friedman_test(df_means)
        print("\nSignificant Friedman Test based on F Distribution => Post Hoc Testing")
        posthoc_paired_comparisons_paper(df_means)

    # f distribution; manually computed
    if friedman_test_f_distribution_paper(df_means) == "rejected":
        print("Significant Friedman Test based on F Distribution from Paper => Post Hoc Testing")
        posthoc_nemenyi_friedman_test(df_means)
        print("\nSignificant Friedman Test based on F Distribution from Paper => Post Hoc Testing")
        posthoc_paired_comparisons_paper(df_means)

    plot_visualization(df_means)
