import numpy as np
import copy
import sys
import time
from datetime import datetime

BIG_NUMBER = 10 ** 10

LOGS_PATH = "logs/"

class PSOAlgorithm:

    def __init__(self, file_path, runs=1, iterations=30, particles=30, inertia_weight=1, type_inertia=0,
                 acceleration_factor_1=2.05, acceleration_factor_2=2.05):

        self.experiment_name = file_path.split("/")[-1]

        self.num_depots, self.num_customers, self.depots_capacities, self.data_matrix = self.read_cost_matrix(file_path=file_path)

        self.num_runs = runs
        self.num_iterations = iterations
        self.inertia = inertia_weight
        self.acc_fac_1 = acceleration_factor_1
        self.acc_fac_2 = acceleration_factor_2

        self.particles_swarm = None  # (dimensions (40, 732))
        self.particles_swarm_dimensions = (particles, self.num_customers + np.sum(self.depots_capacities)) # (40, 733)

        self.personal_best_particles_swarm = None  # (dimensions (40, 732))
        self.global_best_particles_swarm = None  # (dimensions (1, 732))
        self.best_iteration = None

        self.velocities_particles_swarm = None  # (dimensions (40, 732))
        self.min_velocity = -6
        self.max_velocity = 6

        self.eval_value_global_best_particles = None  # (dimension 1))
        self.eval_values_personal_best_particles = []  # (dimension (40,))
        self.eval_value_best_iteration = None

        self.inertia_max = 0.5
        self.inertia_min = 0.2
        self.inertia_diff = self.inertia_max - self.inertia_min
        self.u = 10 ** (np.log(self.num_iterations) - 2)
        self.n = 0.75

        self.inertia_types = [self.iws_constant, self.iws_decline_curve, self.iws_sigmoid_curve]
        self.inertia_type = self.inertia_types[type_inertia]

        self.current_iteration = 0

        self.initial_heuristic_rate = 0.3
        self.grouping_probability = 0.8
        self.divide_number = 10

        self.mutation_rate = 0.01
        self.delta = 3

        self.runs_evaluation_value_global_best = []
        self.runs_evaluation_value_minimum_personal_best = []
        self.runs_evaluation_value_best_iteration = []

########################################################################################################################
# Read data #

    def process_first_line(self, line_str):
        tokens = line_str.split()

        m = int(tokens[0])
        n = int(tokens[1])

        depot_capacities = []
        for index in range(2, len(tokens)):
            depot_capacities += [int(tokens[index])]

        return m, n, depot_capacities

    def process_matrix_line(self, matrix, line_str):
        matrix_line = list(map(lambda t: int(t), line_str.split()))
        return matrix + [matrix_line]

    def read_cost_matrix(self, file_path):
        with open(file_path, "r") as file:
            cost_matrix_list = []

            line = file.readline()
            m, n, depot_capacities = self.process_first_line(line)

            line = file.readline()
            while line:
                cost_matrix_list = self.process_matrix_line(cost_matrix_list, line)
                line = file.readline()

        return m, n, np.array(depot_capacities), np.array(cost_matrix_list)

########################################################################################################################
# Evaluation #

    def decode_particle(self, particle):
        decoding_data = np.arange(self.num_customers)
        decoding_data = np.concatenate((decoding_data, ['seg'] * np.sum(self.depots_capacities)))

        decoding_particle = list(zip(particle, decoding_data))
        sorted_decoding_particle = sorted(decoding_particle, key=lambda x: x[0])

        decoded_particle = []
        for value in sorted_decoding_particle:
            decoded_particle.append(value[1])

        return np.array(decoded_particle)

    def evaluate_decoded_particle(self, decoded_particle):
        cost_particle = 0

        indexes_paths = np.where(decoded_particle == 'seg')[0]
        paths = np.split(decoded_particle, indexes_paths)

        index_depot = 0
        for index_path, path in enumerate(paths):
            if index_path >= np.sum(self.depots_capacities[0: index_depot+1]):
                index_depot += 1

            cost_path = 0

            if len(path) > 0:
                if path[0] == 'seg':
                    path = path[1:].astype(int)
                else:
                    path = path.astype(np.int)

            if len(path) > 0:
                cost_first_trip = self.data_matrix[index_depot][self.num_depots + path[0]]
                cost_last_trip = self.data_matrix[self.num_depots + path[0]][index_depot]

                if cost_first_trip > -1 and cost_last_trip > -1:
                    cost_path = cost_first_trip  # cost plecare din depot la prima locatie
                    cost_path += cost_last_trip  # cost intoarcere in depot de la ultima locatie

                    for pos in range(len(path)-1):
                        cost_trip = self.data_matrix[self.num_depots+path[pos]][self.num_depots+path[pos+1]]
                        if cost_trip != -1:
                            cost_path += cost_trip # cost locatie - locatie
                        else:
                            cost_path = BIG_NUMBER
                            break
                else:
                    cost_path = BIG_NUMBER

            cost_particle += cost_path

        return cost_particle

    def evaluation_function(self, particle):
        particle_decoded = self.decode_particle(particle=particle)
        cost_particle_decoded = self.evaluate_decoded_particle(decoded_particle=particle_decoded)
        return cost_particle_decoded

########################################################################################################################
# Heuristic Approach Initialization #

    def get_node_weight(self, node):
        cost_matrix = self.data_matrix

        in_weight = 0
        for row_index in range(cost_matrix.shape[0]):
            if cost_matrix[row_index, node] != -1:
                in_weight += 1

        out_weight = 0
        for col_index in range(cost_matrix.shape[1]):
            if cost_matrix[node, col_index] != -1:
                out_weight += 1

        return in_weight + out_weight

    def grouping_customers_to_depots(self):
        weighted_values_customers_depots = np.array([self.get_node_weight(val) for val in range(self.num_customers)])
        indexes_sorted_weighted_values = np.argsort(weighted_values_customers_depots)[::-1]

        customers_depot_assignment = []
        for index_customer in range(self.num_customers):
            rand_grouping = np.random.random()

            if rand_grouping < self.grouping_probability:
                customers_depot_assignment.append(indexes_sorted_weighted_values[index_customer] % self.num_depots)
            else:
                customers_depot_assignment.append(np.random.randint(low=0, high=self.num_depots))

        customers_depot_assignment = np.array(customers_depot_assignment)

        grouped_customers = []
        for index_depot in range(self.num_depots):
            customers_indexes = np.where(customers_depot_assignment == index_depot)[0]
            grouped_customers.append(customers_indexes.tolist())

        return np.array(grouped_customers, dtype=object)

    def get_savings_pairs(self, customers, depot_index):
        pairs = []
        values_pairs = []

        for pos1 in range(len(customers) - 1):
            for pos2 in range(pos1+1, len(customers)):
                if self.data_matrix[self.num_depots + customers[pos1]][self.num_depots + customers[pos2]] != -1:
                    pairs.append((customers[pos1], customers[pos2]))
                    value_pair = self.data_matrix[self.num_depots + customers[pos1]][depot_index] + \
                                 self.data_matrix[depot_index][self.num_depots + customers[pos2]] - \
                                 self.data_matrix[self.num_depots + customers[pos1]][self.num_depots + customers[pos2]]  # saving value
                    values_pairs.append(value_pair)

                if self.data_matrix[self.num_depots + customers[pos2]][self.num_depots + customers[pos1]] != -1:
                    pairs.append((customers[pos2], customers[pos1]))
                    value_pair = self.data_matrix[self.num_depots + customers[pos2]][depot_index] + \
                                 self.data_matrix[depot_index][self.num_depots + customers[pos1]] - \
                                 self.data_matrix[self.num_depots + customers[pos2]][self.num_depots + customers[pos1]]  # saving value
                    values_pairs.append(value_pair)

        values_pairs = np.array(values_pairs)
        indexes_sorting = np.argsort(values_pairs)[::-1]

        pairs = np.array(pairs)[indexes_sorting]
        return pairs

    def first_repair_procedure(self, paths):
        current_paths = copy.deepcopy(paths)
        repaired_paths = copy.deepcopy(paths)

        repaired = True
        while repaired == True:
            repaired = False
            for pos1 in range(len(current_paths)):
                for pos2 in range(len(current_paths)):
                    if pos1 != pos2:
                        if self.data_matrix[self.num_depots+current_paths[pos1][-1]][self.num_depots+current_paths[pos2][0]] != -1:
                            if len(repaired_paths) > pos1 and len(repaired_paths) > pos2:
                                if repaired_paths[pos2] == current_paths[pos2]:
                                    repaired_paths[pos1].extend(repaired_paths[pos2])
                                    del repaired_paths[pos2]
                                    repaired = True
            current_paths = copy.deepcopy(repaired_paths)

        return repaired_paths

    def find_first_pair_unused(self, pairs, already_used_customers):
        for index_pair, pair in enumerate(pairs):
            if already_used_customers[pair[0]] == -1 and already_used_customers[pair[1]] == -1:
                return index_pair

        return -1

    def find_path_for_current_pair(self, paths, current_pair):
        for index_path, path in enumerate(paths):
            if path[-1] == current_pair[0]:
                return index_path

        return -1

    def savings_algorithm(self, customers, depot_index):
        paths_found = []
        customer_already_in_path = np.ones((self.num_customers,)) * (-1)

        savings_pairs = self.get_savings_pairs(customers=customers, depot_index=depot_index).tolist()

        index_unused_pair = self.find_first_pair_unused(pairs=savings_pairs, already_used_customers=customer_already_in_path)
        while index_unused_pair > -1:
            paths_found.append(savings_pairs[index_unused_pair])
            customer_already_in_path[savings_pairs[index_unused_pair][0]] = 1
            customer_already_in_path[savings_pairs[index_unused_pair][1]] = 1

            for pair in savings_pairs:
                if customer_already_in_path[pair[0]] == 1 and customer_already_in_path[pair[1]] == -1:
                    index_path = self.find_path_for_current_pair(paths=paths_found, current_pair=pair)
                    if index_path > -1:
                        paths_found[index_path].extend([pair[1]])
                        customer_already_in_path[pair[1]] = 1

            index_unused_pair = self.find_first_pair_unused(pairs=savings_pairs, already_used_customers=customer_already_in_path)

        for pair in savings_pairs:
            if customer_already_in_path[pair[0]] == -1:
                paths_found.append([pair[0]])
                customer_already_in_path[pair[0]] = 1

            if customer_already_in_path[pair[1]] == -1:
                paths_found.append([pair[1]])
                customer_already_in_path[pair[1]] = 1

        if len(paths_found) > self.depots_capacities[depot_index]:
            paths_found = self.first_repair_procedure(paths=paths_found)

        return paths_found

    def verify_path_feasibility(self, current_path, overlimit_paths):
        for index_overlimit_path, overlimit_path in enumerate(overlimit_paths):
            if self.data_matrix[self.num_depots + current_path[-1]][self.num_depots + overlimit_path[0]] != -1:
                return index_overlimit_path, -1
            elif self.data_matrix[self.num_depots + overlimit_path[-1]][self.num_depots + current_path[0]] != -1:
                return index_overlimit_path, 0

        return -1, -1

    def second_repair_procedure(self, overlimit_depot, paths_depots):
        overlimit_paths = copy.deepcopy(paths_depots[overlimit_depot])

        for index_depot, paths_depot in enumerate(paths_depots):
            if index_depot != overlimit_depot: # and len(paths_depot) < self.depots_capacities[index_depot]:
                for index_path, path in enumerate(paths_depot):
                    index_feasibility, position = self.verify_path_feasibility(current_path=path, overlimit_paths=overlimit_paths)

                    if index_feasibility != -1:
                        if position == -1:
                            path += overlimit_paths[index_feasibility]
                        elif position == 0:
                            paths_depots[index_depot][index_path] = overlimit_paths[index_feasibility] + path
                            #paths_depots[index_depot][index_path] = copy.deepcopy(overlimit_paths[index_feasibility].extend(path))

                        del overlimit_paths[index_feasibility]

        return overlimit_paths

    def create_feasible_solution(self, grouped_customers_to_depots):
        feasible_paths_depots = []

        for index, grouped_customers_to_depot in enumerate(grouped_customers_to_depots):
            feasible_paths_depot = self.savings_algorithm(customers=grouped_customers_to_depot, depot_index=index)
            feasible_paths_depots.append(feasible_paths_depot)

        for index_depot, feasible_paths_depot in enumerate(feasible_paths_depots):
            if len(feasible_paths_depot) > self.depots_capacities[index_depot]:
                repaired_feasible_paths_depot = self.second_repair_procedure(overlimit_depot=index_depot, paths_depots=feasible_paths_depots)
                feasible_paths_depots[index_depot] = repaired_feasible_paths_depot

        solution = []
        for index_depot, feasible_paths_depot in enumerate(feasible_paths_depots):
            for path in feasible_paths_depot:
                solution += path + ['seg']

            solution += ['seg'] * (self.depots_capacities[index_depot] - len(feasible_paths_depot))

        return np.array(solution)

    def encode_particle(self, feasible_solution):
        indexes_segmentations = np.where(feasible_solution == 'seg')[0]
        counter_segmentations = self.num_customers

        for index in indexes_segmentations:
            feasible_solution[index] = str(counter_segmentations)
            counter_segmentations += 1

        feasible_solution = np.array(feasible_solution).astype(np.float)
        encoding_data = np.arange(self.particles_swarm_dimensions[1])

        encoding_particle = list(zip(encoding_data, feasible_solution))
        sorted_encoding_particle = sorted(encoding_particle, key=lambda x: x[1])

        encoded_particle = []
        for value in sorted_encoding_particle:
            encoded_particle.append(value[0])

        return np.array(encoded_particle).astype(np.float) / self.divide_number

    def generate_particle(self):
        grouped_customers_to_depots = self.grouping_customers_to_depots()
        feasible_solution = self.create_feasible_solution(grouped_customers_to_depots=grouped_customers_to_depots)
        particle_encoded = self.encode_particle(feasible_solution=feasible_solution)

        return particle_encoded

########################################################################################################################
# Initialization #

    def initialise_particles(self):

        self.particles_swarm = []

        for index in range(self.particles_swarm_dimensions[0]):
            rand_initialization = np.random.random()

            if rand_initialization < self.initial_heuristic_rate:
                new_particle = self.generate_particle()
            else:
                new_particle = np.arange(self.particles_swarm_dimensions[1]).astype(np.float) / self.divide_number
                np.random.shuffle(new_particle)

            self.particles_swarm.append(new_particle)

        self.particles_swarm = np.array(self.particles_swarm) #.astype(np.float)

        self.personal_best_particles_swarm = copy.deepcopy(self.particles_swarm)

        self.global_best_particles_swarm = copy.deepcopy(self.particles_swarm[0])
        self.eval_values_personal_best_particles = self.evaluation_function(particle=self.particles_swarm[0])

        for index_particle in range(1, self.particles_swarm_dimensions[0]):
            eval_value_current_particle = self.evaluation_function(particle=self.particles_swarm[index_particle])

            if eval_value_current_particle < self.eval_values_personal_best_particles:
                self.global_best_particles_swarm = copy.deepcopy(self.particles_swarm[index_particle])
                self.eval_values_personal_best_particles = eval_value_current_particle

    def initialise_velocity(self):
        self.velocities_particles_swarm = np.random.uniform(low=self.min_velocity, high=self.max_velocity,
                                                            size=self.particles_swarm_dimensions)

########################################################################################################################
# Inertia schemes #

    def iws_constant(self):
        return self.inertia

    def iws_decline_curve(self):
        return (1 - self.current_iteration / self.num_iterations) / (1 + self.u * self.current_iteration / self.num_iterations)

    def iws_sigmoid_curve(self):
        return (self.inertia_diff / (1 + np.exp(-self.u * (self.current_iteration - self.n * self.num_iterations)))) + self.inertia_min

########################################################################################################################
# Updating #

    def update_particle_velocity(self, index_particle, inertia_function=None):
        for dimension in range(len(self.particles_swarm[index_particle])):
            rand_1 = np.random.random()
            rand_2 = np.random.random()

            current_velocity = self.velocities_particles_swarm[index_particle][dimension]

            if inertia_function is None:
                inertia_function = self.iws_constant

            updated_velocity = inertia_function() * current_velocity + \
                               self.acc_fac_1 * rand_1 * (
                                       self.personal_best_particles_swarm[index_particle][dimension] -
                                       self.particles_swarm[index_particle][dimension]) + \
                               self.acc_fac_2 * rand_2 * (self.global_best_particles_swarm[dimension] -
                                                          self.particles_swarm[index_particle][dimension])

            self.velocities_particles_swarm[index_particle][dimension] = updated_velocity

    def update_particle_position(self, index_particle):
        self.particles_swarm[index_particle] += self.velocities_particles_swarm[index_particle]

        rand_mutation = np.random.random()
        rand_delta = np.random.uniform(low=(-1)*self.delta, high=self.delta)
        if rand_mutation < self.mutation_rate:
            self.particles_swarm[index_particle] += rand_delta

########################################################################################################################
# Execution #

    def execute_algorithm(self):
        self.write_log_parameters()

        for run in range(self.num_runs):
            if run == 0:
                print(f"Run {run}")
            else:
                print(f"\nRun {run}")

            self.current_iteration = 0

            self.write_number_run(run)

            self.initialise_particles()
            self.initialise_velocity()

            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                start = time.time_ns()
            else:
                start = time.time()

            self.eval_values_personal_best_particles = []
            for index_particle in range(self.particles_swarm_dimensions[0]):
                self.eval_values_personal_best_particles.append(
                    self.evaluation_function(particle=self.personal_best_particles_swarm[index_particle]))

            self.eval_value_global_best_particles = self.evaluation_function(particle=self.global_best_particles_swarm)

            for iteration in range(self.num_iterations):

                self.current_iteration = iteration

                self.best_iteration = copy.deepcopy(self.particles_swarm[0])
                self.eval_value_best_iteration = self.evaluation_function(particle=self.best_iteration)
                self.index_particle_best_iteration = 0

                for index_particle in range(self.particles_swarm_dimensions[0]):

                    self.update_particle_velocity(index_particle)
                    self.update_particle_position(index_particle)

                    eval_value_current_particle = self.evaluation_function(particle=self.particles_swarm[index_particle])

                    if eval_value_current_particle < self.eval_values_personal_best_particles[index_particle]:
                        self.personal_best_particles_swarm[index_particle] = copy.deepcopy(self.particles_swarm[index_particle])
                        self.eval_values_personal_best_particles[index_particle] = eval_value_current_particle

                        if self.eval_value_best_iteration > eval_value_current_particle:
                            self.eval_value_best_iteration = eval_value_current_particle
                            self.index_particle_best_iteration = index_particle
                            self.best_iteration = copy.deepcopy(self.particles_swarm[index_particle])

                for index_particle in range(self.particles_swarm_dimensions[0]):
                    if self.eval_values_personal_best_particles[index_particle] < self.eval_value_global_best_particles:
                        self.global_best_particles_swarm = copy.deepcopy(self.personal_best_particles_swarm[index_particle])
                        self.eval_value_global_best_particles = self.eval_values_personal_best_particles[index_particle]

                if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                    end = time.time_ns()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")
                else:
                    end = time.time()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start)} seconds.")

                self.write_log_info_iteration(iteration)

            self.write_log_info_run(run)

        self.write_log_info_runs()

########################################################################################################################
# Logging #

    def write_log_parameters(self):
        with open(LOGS_PATH + self.experiment_name[:-4] + "_parameters.txt", "w") as file:
            parameters = f"{self.experiment_name[:-4]}\n" \
                         + f"number_particles={self.particles_swarm_dimensions[0]}\n" \
                         + f"runs={self.num_runs}\n" \
                         + f"iterations={self.num_iterations}\n" \
                         + f"inertia_weight={self.inertia}\n" \
                         + f"acceleration_factor_1={self.acc_fac_1}\n" \
                         + f"acceleration_factor_2={self.acc_fac_2}\n" \
                         + f"minimum_velocity={self.min_velocity}\n" \
                         + f"maximum_velocity={self.max_velocity}\n" \
                         + f"inertia_type={self.inertia_type}\n"

            file.write(parameters)

        with open(LOGS_PATH + self.experiment_name[:-4] + "_iterations.txt", "w") as file:
            pass

        with open(LOGS_PATH + self.experiment_name[:-4] + "_runs.txt", "w") as file:
            pass

    def write_number_run(self, number_run):
        with open(LOGS_PATH + self.experiment_name[:-4] + "_iterations.txt", "a+") as file:
            if number_run == 0:
                file.write(f"Run {number_run} \n\n")
            else:
                file.write(f"\nRun {number_run} \n\n")

    def get_log_info(self):
        self.evaluation_value_global_best = self.eval_value_global_best_particles

        self.evaluation_value_minimum_personal_best = np.min(self.eval_values_personal_best_particles)
        self.number_particle_minimum_personal_best = np.argmin(self.eval_values_personal_best_particles)

        self.evaluation_value_best_iteration = self.eval_value_best_iteration
        self.number_particle_minimum_best_iteration = self.index_particle_best_iteration

        delimiter = " ;" + " " * 4

        info = f"value_global_best = {self.evaluation_value_global_best}{delimiter}"
        info += f"number_particle_minimum_personal_best = {self.number_particle_minimum_personal_best}{delimiter}"
        info += f"value_minimum_personal_best = {self.evaluation_value_minimum_personal_best}{delimiter}"
        info += f"number_particle_minimum_best_iteration = {self.number_particle_minimum_best_iteration}{delimiter}"
        info += f"value_best_iteration = {self.evaluation_value_best_iteration}{delimiter}"

        return info

    def write_log_info_iteration(self, number_iteration):
        info_iteration = f"Iteration {number_iteration}: "
        info_iteration += self.get_log_info()

        with open(LOGS_PATH + self.experiment_name[:-4] + "_iterations.txt", "a+") as file:
            file.write(info_iteration + "\n")

    def write_log_info_run(self, number_run):
        info_run = f"Run {number_run}: "
        info_run += self.get_log_info()

        with open(LOGS_PATH + self.experiment_name[:-4] + "_runs.txt", "a+") as file:
            file.write(info_run + "\n")

        self.runs_evaluation_value_global_best.append(self.evaluation_value_global_best)
        self.runs_evaluation_value_minimum_personal_best.append(self.evaluation_value_minimum_personal_best)
        self.runs_evaluation_value_best_iteration.append(self.evaluation_value_best_iteration)

    def write_log_info_runs(self):
        info_runs = f"\n\nRuns: "

        delimiter = " ;" + " " * 4

        info_runs += f"min_global_best = {np.min(self.runs_evaluation_value_global_best)}{delimiter}"
        info_runs += f"max_global_best = {np.max(self.runs_evaluation_value_global_best)}{delimiter}"
        info_runs += f"mean_global_best = {np.mean(self.runs_evaluation_value_global_best)}{delimiter}"
        info_runs += f"std_global_best = {np.std(self.runs_evaluation_value_global_best)}{delimiter}"

        info_runs += f"\n      "
        info_runs += f"min_minimum_personal_best = {np.min(self.runs_evaluation_value_minimum_personal_best)}{delimiter}"
        info_runs += f"max_minimum_personal_best = {np.max(self.runs_evaluation_value_minimum_personal_best)}{delimiter}"
        info_runs += f"mean_minimum_personal_best = {np.mean(self.runs_evaluation_value_minimum_personal_best)}{delimiter}"
        info_runs += f"std_minimum_personal_best = {np.std(self.runs_evaluation_value_minimum_personal_best)}{delimiter}"

        info_runs += f"\n      "
        info_runs += f"min_best_iteration = {np.min(self.runs_evaluation_value_best_iteration)}{delimiter}"
        info_runs += f"max_best_iteration = {np.max(self.runs_evaluation_value_best_iteration)}{delimiter}"
        info_runs += f"mean_best_iteration = {np.mean(self.runs_evaluation_value_best_iteration)}{delimiter}"
        info_runs += f"std_best_iteration = {np.std(self.runs_evaluation_value_best_iteration)}{delimiter}"

        with open(LOGS_PATH + self.experiment_name[:-4] + "_runs.txt", "a+") as file:
            file.write(info_runs + "\n")
