def row_sum_odd_numbers(n):
    rows = n
    odd = 1
    for i in range(1, rows + 1):
        row = []
        for j in range(i):
            row.append(odd)
            odd += 2
    return sum(row)

    
print(row_sum_odd_numbers(6))