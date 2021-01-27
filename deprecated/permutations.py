from conjecture.knuth_partition import algorithm_u as set_partition
from itertools import permutations, product


def get_permutations_helper(machine_order):
    perms = list(permutations(machine_order))
    [list(perm) for perm in perms]
    return [list(perm) for perm in perms]


def get_permutations(task_list, num_machines):
    # Get partitions
    set_partitions = list(set_partition(task_list, num_machines))

    # Permute across each partition
    task_permutations = []
    for partition in set_partitions:
        partition_permutations = [[list(machine)] if (len(list(machine)) == 0 or len(list(machine)) == 1) else get_permutations_helper(machine) for machine in partition]
        task_permutations.extend(list(product(*partition_permutations)))

    task_permutations = [list(perm) for perm in task_permutations]
    return task_permutations


if __name__ == "__main__":
    print(get_permutations([1, 2, 3, 4], 3))
    set_partitions = list(set_partition([1, 2, 3, 4], 3))
    a = set_partitions[0]
    print(len(list(permutations(a[2]))))

