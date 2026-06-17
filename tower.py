def tower_builder(n_floors):
    tower = []

    width = 2 * n_floors -1

    for floor in range(1, n_floors + 1):
        stars = '*' * (2* floor - 1)
        tower.append(stars.center(width))

    return tower

print(tower_builder(6))