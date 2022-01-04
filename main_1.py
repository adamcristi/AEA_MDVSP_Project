import sys
import time

from pso.pso_algorithm import PSOAlgorithm

# Particle Swarm Optimization Algorithm

if __name__ == "__main__":
    files = ["m8n500s0.inp",
             "m8n500s1.inp",
             "m8n500s2.inp",
             "m8n500s3.inp",
             "m8n500s4.inp",

             "m8n1000s0.inp",
             "m8n1000s1.inp",
             "m8n1000s2.inp",
             "m8n1000s3.inp",
             "m8n1000s4.inp",

             "m8n1500s0.inp",
             "m8n1500s1.inp",
             "m8n1500s2.inp",
             "m8n1500s3.inp",
             "m8n1500s4.inp"]

    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        start = time.time_ns()
    else:
        start = time.time()

    for file in files:
        print(file)
        pso = PSOAlgorithm(file_path="data/{}".format(file),
                           runs=3,
                           iterations=1000,
                           particles=40,
                           inertia_weight=0.5,
                           type_inertia=2,
                           acceleration_factor_1=2.05,
                           acceleration_factor_2=2.05)

        pso.execute_algorithm()
        print()
        #break

    if sys.version_info.major == 3 and sys.version_info.minor >= 7:
        end = time.time_ns()
        print(f"Total time: {(end-start) / 1e9} seconds.")
    else:
        end = time.time()
        print(f"Total time: {(end - start)} seconds.")

